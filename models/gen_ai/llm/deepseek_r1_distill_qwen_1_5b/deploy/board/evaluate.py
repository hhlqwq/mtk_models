"""在开发板使用常驻 Neuron 硬件 Runtime 执行分片 LLM 与评测."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np


def available_memory_mib():
    """读取系统可用内存,补充 RSS 未覆盖的驱动分配内存证据."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("无法读取 MemAvailable.")


def verify_artifacts(models):
    """加载前核对板端实际文件,防止上传不完整或混用旧分片."""
    manifest = json.loads((models / "artifact_manifest.json").read_text())
    for entry in manifest["files"]:
        print(f"[产物校验] {entry['name']}", flush=True)
        path = models / entry["name"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"部署文件校验失败: {path}")


class NeuronGraph:
    """绑定一个 DLA,严格校验原生 FP16 缓冲区契约."""

    def __init__(self, library, path, input_shapes, output_shapes):
        """加载 DLA 并核对全部 I/O 数量、大小."""
        self.library = library
        self.runtime = ctypes.c_void_p()
        self.check(library.CreateModel(str(path).encode(),
                                       ctypes.byref(self.runtime)))
        self.inputs = [np.zeros(shape, dtype=np.float16)
                       for shape in input_shapes]
        self.outputs = [np.zeros(shape, dtype=np.float16)
                        for shape in output_shapes]
        try:
            inputs = ctypes.c_size_t()
            outputs = ctypes.c_size_t()
            self.check(library.ModelCounts(self.runtime, ctypes.byref(inputs),
                                           ctypes.byref(outputs)))
            if (inputs.value, outputs.value) != (len(self.inputs),
                                                 len(self.outputs)):
                raise ValueError(f"DLA I/O 数量不匹配: {path}")
            for output, buffers in enumerate((self.inputs, self.outputs)):
                for index, buffer in enumerate(buffers):
                    size = ctypes.c_size_t()
                    self.check(library.ModelSize(self.runtime, index, output,
                                                 ctypes.byref(size)))
                    if size.value != buffer.nbytes:
                        raise ValueError(f"DLA 原生布局不匹配: {path},"
                                         f"output={output},index={index},"
                                         f"实际={size.value},预期={buffer.nbytes}")
        except Exception:
            self.close()
            raise

    @staticmethod
    def check(code):
        """将 Runtime 返回码转换为明确的失败,不回退 CPU."""
        if code != 0:
            raise RuntimeError(f"Neuron 硬件推理失败,code={code}")

    def run(self, inputs):
        """复制输入并执行一次 NPU 调用,返回输出与纯调用耗时."""
        for target, source in zip(self.inputs, inputs, strict=True):
            np.copyto(target, source, casting="unsafe")
        input_pointers = (ctypes.c_void_p * len(self.inputs))(
            *[buffer.ctypes.data for buffer in self.inputs])
        output_pointers = (ctypes.c_void_p * len(self.outputs))(
            *[buffer.ctypes.data for buffer in self.outputs])
        input_bytes = (ctypes.c_size_t * len(self.inputs))(
            *[buffer.nbytes for buffer in self.inputs])
        output_bytes = (ctypes.c_size_t * len(self.outputs))(
            *[buffer.nbytes for buffer in self.outputs])
        start = time.perf_counter()
        self.check(self.library.RunModel(
            self.runtime, input_pointers, input_bytes, len(self.inputs),
            output_pointers, output_bytes, len(self.outputs)))
        return self.outputs, (time.perf_counter() - start) * 1000

    def close(self):
        """显式释放 Runtime,包括异常后的资源清理."""
        if self.runtime.value:
            self.library.ReleaseModel(self.runtime)
            self.runtime = ctypes.c_void_p()


def load_library(path):
    """声明桥接函数类型,防止 AArch64 指针被隐式截断."""
    library = ctypes.CDLL(str(path))
    pointer = ctypes.c_void_p
    size = ctypes.c_size_t
    library.CreateModel.argtypes = [ctypes.c_char_p, ctypes.POINTER(pointer)]
    library.ModelCounts.argtypes = [pointer, ctypes.POINTER(size),
                                   ctypes.POINTER(size)]
    library.ModelSize.argtypes = [pointer, size, ctypes.c_int,
                                 ctypes.POINTER(size)]
    library.RunModel.argtypes = [pointer, ctypes.POINTER(pointer),
                                ctypes.POINTER(size), size,
                                ctypes.POINTER(pointer), ctypes.POINTER(size), size]
    library.ReleaseModel.argtypes = [pointer]
    library.ReleaseModel.restype = None
    return library


class Decoder:
    """常驻分片模型,在 CPU 管理 Token/KV,在 NPU 执行模型算子."""

    def __init__(self, models, library):
        """加载元数据、Embedding 与全部 DLA."""
        self.spec = json.loads((models / "export_manifest.json").read_text())
        if self.spec["probe_only"]:
            raise ValueError("算子探测模型不能用于正式评测.")
        spec = self.spec
        self.embedding = np.memmap(models / "embedding_fp16.bin", mode="r",
                                   dtype="<f2", shape=(spec["vocab"], spec["hidden"]))
        hidden = (1, 1, spec["hidden"])
        cache = (1, spec["kv_heads"], spec["context"] - 1, spec["head_dim"])
        angle = (1, 1, 1, spec["head_dim"])
        self.layers = []
        self.heads = []
        try:
            for graph in spec["graphs"]:
                print(f"[加载] {graph['name']}", flush=True)
                if graph["kind"] == "decoder":
                    module = NeuronGraph(
                        library, models / f"{graph['name']}.dla",
                        [hidden, cache, cache, angle, angle,
                         (1, 1, 1, spec["context"])],
                        [hidden, (1, spec["kv_heads"], 1, spec["head_dim"]),
                         (1, spec["kv_heads"], 1, spec["head_dim"])])
                    self.layers.append(module)
                else:
                    module = NeuronGraph(library, models / f"{graph['name']}.dla",
                                         [hidden], [(1, 1, graph["count"])])
                    self.heads.append((graph, module))
        except Exception:
            self.close()
            raise
        self.reset()
        self.minimum_available_mib = available_memory_mib()

    def reset(self):
        """清零样本间 KV Cache,防止上下文串扰."""
        spec = self.spec
        self.cache = [np.zeros((2, 1, spec["kv_heads"], spec["context"] - 1,
                               spec["head_dim"]), dtype=np.float16)
                      for _ in self.layers]
        self.position = 0

    def step(self, token):
        """执行一个 Token,拒绝超出固定上下文容量的输入."""
        spec = self.spec
        if self.position >= spec["context"] or not 0 <= token < spec["vocab"]:
            raise ValueError("Token 或上下文越界.")
        hidden = self.embedding[token].reshape(1, 1, -1)
        frequencies = spec["rope_theta"] ** (
            -np.arange(0, spec["head_dim"], 2, dtype=np.float32) / spec["head_dim"])
        angles = np.tile(frequencies * self.position, 2).reshape(1, 1, 1, -1)
        mask = np.full((1, 1, 1, spec["context"]), -10000, dtype=np.float16)
        mask[..., spec["context"] - 1 - self.position:] = 0
        npu_ms = 0.0
        for layer_index, (graph, cache) in enumerate(
                zip(self.layers, self.cache, strict=True)):
            outputs, duration = graph.run(
                [hidden, cache[0], cache[1], np.cos(angles), np.sin(angles), mask])
            hidden = outputs[0]
            if any(not np.isfinite(output).all() for output in outputs):
                raise RuntimeError(f"第 {layer_index} 层在位置 {self.position}"
                                   "产生非有限输出.")
            for index in (0, 1):
                cache[index, :, :, :-1, :] = cache[index, :, :, 1:, :].copy()
                cache[index, :, :, -1:, :] = outputs[index + 1]
            npu_ms += duration
        logits = np.empty(spec["vocab"], dtype=np.float32)
        for info, graph in self.heads:
            outputs, duration = graph.run([hidden])
            start = info["start"]
            logits[start:start + info["count"]] = outputs[0].reshape(-1)
            npu_ms += duration
        if not np.isfinite(logits).all():
            raise RuntimeError("模型产生非有限 Logits.")
        self.position += 1
        self.minimum_available_mib = min(self.minimum_available_mib,
                                         available_memory_mib())
        return logits, npu_ms

    def close(self):
        """释放全部已加载分片."""
        for graph in self.layers:
            graph.close()
        for _, graph in self.heads:
            graph.close()


def evaluate(args):
    """运行固定样例并保存真实输出、教师强制 NLL 和性能证据."""
    args.output.mkdir(parents=True, exist_ok=False)
    samples = json.loads(args.samples.read_text())
    start = time.perf_counter()
    decoder = None
    predictions = []
    baseline_available_mib = available_memory_mib()
    try:
        verify_artifacts(args.models)
        verification_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        decoder = Decoder(args.models, load_library(args.library))
        load_ms = (time.perf_counter() - start) * 1000
        loaded_available_mib = available_memory_mib()
        for index, sample in enumerate(samples["samples"], 1):
            print(f"[板端 {index}/{len(samples['samples'])}] {sample['id']}", flush=True)
            decoder.reset()
            start = time.perf_counter()
            npu_ms = 0.0
            for token in sample["input_ids"]:
                logits, duration = decoder.step(token)
                npu_ms += duration
            first_ms = (time.perf_counter() - start) * 1000
            prefill_npu_ms = npu_ms
            tokens = []
            nll = []
            argmax = []
            decode_start = time.perf_counter()
            targets = sample.get("target_ids", [])
            limit = len(targets) if targets else samples["max_new_tokens"]
            reached_eos = False
            for offset in range(limit):
                prediction = int(np.argmax(logits))
                argmax.append(prediction)
                if targets:
                    token = targets[offset]
                    maximum = float(logits.max())
                    nll.append(float(np.log(np.exp(logits.astype(np.float64) - maximum).sum())
                                     + maximum - logits[token]))
                else:
                    token = prediction
                tokens.append(token)
                if not targets and token in samples["eos_ids"]:
                    reached_eos = True
                    break
                if offset + 1 < limit:
                    logits, duration = decoder.step(token)
                    npu_ms += duration
                if (offset + 1) % 16 == 0:
                    print(f"[Token] {sample['id']}: {offset + 1}/{limit}", flush=True)
            decode_ms = (time.perf_counter() - decode_start) * 1000
            predictions.append({"id": sample["id"], "tokens": tokens,
                                "argmax": argmax, "nll": nll, "reached_eos": reached_eos,
                                "ttft_ms": first_ms, "decode_ms": decode_ms,
                                "prefill_tokens": len(sample["input_ids"]),
                                "prefill_npu_ms": prefill_npu_ms,
                                "npu_ms": npu_ms,
                                "end_to_end_ms": (time.perf_counter() - start) * 1000})
            (args.output / "predictions.json").write_text(
                json.dumps(predictions, ensure_ascii=False, indent=2))
        summary = {"run_id": args.run_id, "backend": "Neuron MDLA hardware",
                   "system": platform.platform(), "python": platform.python_version(),
                   "numpy": np.__version__,
                   "precision": "FP16 DLA", "load_ms": load_ms,
                   "artifact_integrity_verified": True,
                   "artifact_verification_ms": verification_ms,
                   "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   "baseline_system_available_mib": baseline_available_mib,
                   "loaded_system_available_mib": loaded_available_mib,
                   "minimum_system_available_mib": decoder.minimum_available_mib,
                   "system_available_decrease_mib": baseline_available_mib - decoder.minimum_available_mib,
                   "sample_count": len(predictions), "status": "board_inference_executed",
                   "evaluation_scope": samples["evaluation_scope"]}
        (args.output / "board_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"run_id": args.run_id, "status": "failed",
                        "error": str(error), "completed_samples": len(predictions)},
                       ensure_ascii=False, indent=2))
        raise
    finally:
        if decoder is not None:
            decoder.close()


def parse_args():
    """解析部署路径、固定测试输入及独立运行编号."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
