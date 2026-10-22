"""通过 SDK C ABI 直接执行 Neuron 硬件推理,不依赖自编桥接库."""

import ctypes
import csv
import os
from pathlib import Path
import resource
import time

import numpy as np


class EnvOptions(ctypes.Structure):
    """对应 SDK RuntimeAPI.h 的硬件运行选项,保留自然对齐."""

    _fields_ = [("deviceKind", ctypes.c_uint32), ("MDLACoreOption", ctypes.c_int),
                ("CPUThreadNum", ctypes.c_uint8),
                ("suppressInputConversion", ctypes.c_bool),
                ("suppressOutputConversion", ctypes.c_bool)]


class BufferAttribute(ctypes.Structure):
    """对应 Types.h 的普通内存缓冲区属性."""

    _fields_ = [("ionFd", ctypes.c_int)]


class Runtime:
    """常驻加载 DLA,按实际 SDK 字节数绑定 NumPy 缓冲区."""

    def __init__(self, model):
        """按已核对的 SDK 8 C ABI 创建硬件实例并读取 I/O 大小."""
        self.library = ctypes.CDLL(os.environ.get(
            "MTK_RUNTIME_LIBRARY", "libneuronusdk_runtime.mtk.so.8"))
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        signatures = {
            "create": [ctypes.POINTER(EnvOptions), ctypes.POINTER(pointer)],
            "loadNetworkFromFile": [pointer, ctypes.c_char_p],
            "getInputNumber": [pointer, ctypes.POINTER(size)],
            "getOutputNumber": [pointer, ctypes.POINTER(size)],
            "getInputPaddedSize": [pointer, ctypes.c_uint64, ctypes.POINTER(size)],
            "getOutputPaddedSize": [pointer, ctypes.c_uint64, ctypes.POINTER(size)],
            "setInput": [pointer, ctypes.c_uint64, pointer, size, BufferAttribute],
            "setOutput": [pointer, ctypes.c_uint64, pointer, size, BufferAttribute],
            "inference": [pointer], "release": [pointer],
        }
        for name, arguments in signatures.items():
            function = getattr(self.library, "NeuronRuntime_" + name)
            function.argtypes = arguments
            function.restype = None if name == "release" else ctypes.c_int
        self.handle = pointer()
        options = EnvOptions(4, 0, 1, True, True)
        self.call("create", ctypes.byref(options), ctypes.byref(self.handle))
        try:
            self.call("loadNetworkFromFile", self.handle,
                      os.fsencode(Path(model).resolve()))
            self.sizes = []
            for kind in ("Input", "Output"):
                count = size()
                self.call(f"get{kind}Number", self.handle, ctypes.byref(count))
                sizes = []
                for index in range(count.value):
                    length = size()
                    self.call(f"get{kind}PaddedSize", self.handle, index,
                              ctypes.byref(length))
                    if length.value <= 0:
                        raise ValueError("SDK 返回空张量.")
                    sizes.append(length.value)
                self.sizes.append(sizes)
        except Exception:
            self.close()
            raise

    def call(self, name, *arguments):
        """保留 SDK 操作和错误码,硬件失败时直接停止."""
        code = getattr(self.library, "NeuronRuntime_" + name)(*arguments)
        if code != 0:
            raise RuntimeError(f"NeuronRuntime_{name} 失败: {code}.")

    def tensor(self, index, shape, dtype, *, output=False, fill=0, padded=True):
        """创建原生存储及逻辑视图,只接受连续或末维 16 字节补齐布局."""
        dtype = np.dtype(dtype)
        shape = tuple(shape)
        rows, width = int(np.prod(shape[:-1])), shape[-1]
        size = self.sizes[int(output)][index]
        plain = rows * width * dtype.itemsize
        stride = ((width * dtype.itemsize + 15) // 16 * 16) // dtype.itemsize
        if size != plain and (not padded or size != rows * stride * dtype.itemsize):
            raise ValueError(f"DLA 张量大小不符: output={output}, index={index}, "
                             f"logical={plain}, actual={size}.")
        storage = np.full(size // dtype.itemsize, fill, dtype=dtype)
        view = storage.reshape(rows, -1)[:, :width].reshape(shape)
        return storage, view

    def infer(self, inputs, outputs):
        """每次重新绑定实际缓冲区,仅计时 SDK 同步 inference 调用."""
        if not self.handle:
            raise RuntimeError("Runtime 已关闭.")
        for kind, buffers, expected in zip(("Input", "Output"),
                                            (inputs, outputs), self.sizes):
            if len(buffers) != len(expected):
                raise ValueError(f"DLA {kind} 数量不符.")
            for index, (buffer, size) in enumerate(zip(buffers, expected)):
                if not buffer.flags.c_contiguous or buffer.nbytes != size:
                    raise ValueError(f"DLA {kind}[{index}] 缓冲区不符.")
                self.call(f"set{kind}", self.handle, index,
                          ctypes.c_void_p(buffer.ctypes.data), size, BufferAttribute(-1))
        started = time.perf_counter()
        self.call("inference", self.handle)
        return (time.perf_counter() - started) * 1000

    def close(self):
        """释放当前实例,不修改系统库或其他进程."""
        if self.handle:
            self.library.NeuronRuntime_release(self.handle)
            self.handle = ctypes.c_void_p()

    def __enter__(self):
        """在限定作用域中复用已加载的模型."""
        return self

    def __exit__(self, *_):
        """正常和异常退出均释放 SDK 实例."""
        self.close()


def quantize(values, scale, zero_point, dtype=np.int8):
    """按模型量化契约取整并饱和,不改变预处理协议."""
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(values).all():
        raise ValueError("量化参数或输入数值无效.")
    limits = np.iinfo(dtype)
    return np.clip(np.rint(values / scale) + zero_point,
                   limits.min, limits.max).astype(dtype)


def peak_rss_kib():
    """读取 Linux 当前 Python 进程的峰值常驻内存,单位 KiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def read_contract(path, names, shapes):
    """读取既有七列 INT8 张量契约,验证语义、输出顺序与形状."""
    with Path(path).open(encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    tensors = {}
    for row in rows:
        name, index, channels, height, width, scale, zero = row
        item = {"name": name, "index": int(index),
                "shape": [1, int(channels), int(height), int(width)],
                "scale": float(scale), "zero_point": int(zero)}
        if (name in tensors or not np.isfinite(item["scale"]) or item["scale"] <= 0
                or not -128 <= item["zero_point"] <= 127):
            raise ValueError(f"无效张量契约: {row}.")
        tensors[name] = item
    if set(tensors) != {"input", *names} or tensors["input"]["shape"] != [1, 3, 640, 640]:
        raise ValueError("张量契约缺少输入输出或输入形状错误.")
    for name, shape in zip(names, shapes):
        if tensors[name]["shape"] != shape:
            raise ValueError(f"输出形状不符: {name}.")
    outputs = sorted((tensors[name] for name in names), key=lambda item: item["index"])
    if [item["index"] for item in outputs] != list(range(len(names))):
        raise ValueError("输出索引必须唯一且连续.")
    return tensors["input"], outputs
