"""使用真实 Neuron MDLA 硬件执行统一 WikiText2 子集 PPL."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha256(path):
    """分块计算实际板端资源哈希."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(code):
    """遇到硬件 Runtime 错误直接停止,不回退到 CPU."""
    if code != 0:
        raise RuntimeError(f"Neuron 硬件 Runtime 错误: {code}")


class HardwareModel:
    """绑定实际 INT16 缓冲区,使用既有 Neuron C++ 桥接接口."""

    def __init__(self, library_path, model_path, contract):
        """声明 AArch64 接口类型并核对每个实际缓冲区大小."""
        self.library = ctypes.CDLL(str(library_path.resolve()))
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        self.library.CreateModel.argtypes = [ctypes.c_char_p, ctypes.POINTER(pointer)]
        self.library.ModelCounts.argtypes = [pointer, ctypes.POINTER(size),
                                            ctypes.POINTER(size)]
        self.library.ModelSize.argtypes = [pointer, size, ctypes.c_int, ctypes.POINTER(size)]
        self.library.RunModel.argtypes = [pointer, ctypes.POINTER(pointer),
                                         ctypes.POINTER(size), size,
                                         ctypes.POINTER(pointer), ctypes.POINTER(size), size]
        self.library.ReleaseModel.argtypes = [pointer]
        self.library.ReleaseModel.restype = None
        self.runtime = pointer()
        check(self.library.CreateModel(str(model_path.resolve()).encode(),
                                        ctypes.byref(self.runtime)))
        self.inputs = [np.full(item["shape"], item["zero_point"], dtype="<i2")
                       for item in contract["inputs"]]
        self.outputs = [np.zeros(item["shape"], dtype="<i2") for item in contract["outputs"]]
        input_count, output_count = size(), size()
        try:
            check(self.library.ModelCounts(self.runtime, ctypes.byref(input_count),
                                            ctypes.byref(output_count)))
            if (input_count.value, output_count.value) != (len(self.inputs), len(self.outputs)):
                raise ValueError("DLA I/O 数量与实际 TFLite 契约不符.")
            for output, buffers in enumerate((self.inputs, self.outputs)):
                for index, buffer in enumerate(buffers):
                    actual = size()
                    check(self.library.ModelSize(self.runtime, index, output, ctypes.byref(actual)))
                    if actual.value != buffer.nbytes:
                        raise ValueError(f"DLA 缓冲区大小不符: output={output},index={index}")
        except Exception:
            self.close()
            raise

    def run(self):
        """同步执行一次真实硬件推理并返回原始输出."""
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        inputs = (pointer * len(self.inputs))(*(b.ctypes.data for b in self.inputs))
        outputs = (pointer * len(self.outputs))(*(b.ctypes.data for b in self.outputs))
        input_sizes = (size * len(self.inputs))(*(b.nbytes for b in self.inputs))
        output_sizes = (size * len(self.outputs))(*(b.nbytes for b in self.outputs))
        check(self.library.RunModel(self.runtime, inputs, input_sizes, len(self.inputs),
                                    outputs, output_sizes, len(self.outputs)))
        return self.outputs

    def close(self):
        """释放本次模型 Runtime,保留其他进程和系统库."""
        if self.runtime:
            self.library.ReleaseModel(self.runtime)
            self.runtime = ctypes.c_void_p()


def evaluate(args):
    """逐块重置上下文并评分实际 MDLA logits,保留 Token NLL 和运行时间."""
    protocol_path = args.quality_data / "board_quality_protocol.json"
    if not protocol_path.exists():
        protocol_path = args.quality_data / "tflite_quality.json"
    report = json.loads(protocol_path.read_text())
    contract = json.loads((args.quality_data / "board_input_contract.json").read_text())
    manifest = json.loads((args.package / "manifest.json").read_text())
    if report["protocol"] != "fixed_text_rows_v1" or report["block_tokens"] != 128:
        raise ValueError("需要统一的原始文本评价协议.")
    if contract["tflite_sha256"] != report["tflite_sha256"]:
        raise ValueError("输入契约与主机质量报告不符.")
    inputs_path = args.quality_data / "board_inputs.npz"
    if sha256(inputs_path) != contract["board_inputs_sha256"]:
        raise ValueError("SDK 输入资源哈希不符.")
    model_path = args.package / f"{manifest['context']}c/prompt.dla"
    embedding_path = args.package / "tokenizer/embedding_int16.bin"
    for path in (model_path, embedding_path):
        name = path.relative_to(args.package).as_posix()
        entry = next(item for item in manifest["files"] if item["name"] == name)
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"交付资源哈希不符: {name}")
    if sha256(embedding_path) != report["embedding_sha256"]:
        raise ValueError("主机与板端 embedding 不同.")
    width = contract["inputs"][0]["shape"][-1]
    embedding = np.memmap(embedding_path, dtype="<i2", mode="r").reshape(-1, width)
    if contract["inputs"][0]["name"] != "input_embeds" or \
            contract["outputs"][0]["shape"] != [1, 128, len(embedding)]:
        raise ValueError("图的 embedding 或词表 logits 接口不符.")
    token_blocks = np.array(report["input_ids"], dtype=np.int64).reshape(-1, 128)
    model = HardwareModel(args.library, model_path, contract)
    sdk_inputs = np.load(inputs_path)
    for index, item in enumerate(contract["inputs"]):
        if item["name"] in ("mask", "pos_emb"):
            np.copyto(model.inputs[index], sdk_inputs[item["name"]])
    output_spec = contract["outputs"][0]
    rows, nll_values = [], []
    started = time.perf_counter()
    try:
        for index, tokens in enumerate(token_blocks):
            np.copyto(model.inputs[0], embedding[tokens][None, :, :])
            tick = time.perf_counter()
            raw = model.run()[0][0, :-1, :]
            inference_seconds = time.perf_counter() - tick
            logits = (raw.astype(np.float64) - output_spec["zero_point"]) * output_spec["scale"]
            maximum = logits.max(axis=-1)
            normalizer = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=-1))
            nll = normalizer - logits[np.arange(127), tokens[1:]]
            nll_values.extend(nll.tolist())
            rows.append({"block": index, "inference_seconds": inference_seconds})
            print(f"[MDLA PPL] {index + 1}/{len(token_blocks)}", flush=True)
    finally:
        model.close()
    result = {key: report[key] for key in ("protocol", "model_name", "corpus_sha256",
              "selected_text_sha256", "blocks", "block_tokens", "scored_tokens",
              "excluded_tail_tokens", "input_ids", "reset_context_each_block")}
    result.update(backend="neuron_mdla_hardware", scope="board_fixed_text_subset_not_full_benchmark",
                  perplexity=float(np.exp(np.mean(nll_values))), nll=nll_values,
                  elapsed_seconds=time.perf_counter() - started, block_timings=rows,
                  dla_sha256=sha256(model_path), embedding_sha256=sha256(embedding_path),
                  bridge_sha256=sha256(args.library), cpu_fallback=False)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "npu_quality.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"[MDLA 质量结果] PPL={result['perplexity']:.6f}", flush=True)


def main():
    """解析已验证交付包、SDK 输入契约和既有硬件桥接库."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--quality-data", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
