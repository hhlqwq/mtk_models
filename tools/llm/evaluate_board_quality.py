"""使用真实 Neuron MDLA 硬件执行统一 WikiText2 子集 PPL."""

import argparse
import ctypes
import json
from pathlib import Path
import time

import numpy as np

from tools.files import sha256

from tools.evaluation import (
    add_test_argument, is_test, record_selection, result_status, scope_metadata,
)


def check(code):
    """遇到硬件 Runtime 错误直接停止,不回退到 CPU."""
    if code != 0:
        raise RuntimeError(f"Neuron 硬件 Runtime 错误: {code}")


def transfer_cache(source_contract, source_outputs, target_contract, target_inputs,
                   append=False):
    """按真实量化参数转移 KV,与 SDK 后缀有效缓存布局保持一致."""
    outputs = {item["name"]: (item, value) for item, value in
               zip(source_contract["outputs"], source_outputs)}
    for item, target in zip(target_contract["inputs"], target_inputs):
        if not item["name"].startswith("past_"):
            continue
        source, raw = outputs[item["name"].replace("past_", "curr_", 1)]
        if raw.ndim != 4 or target.ndim != 4 or raw.shape[:2] != target.shape[:2] or \
                raw.shape[3] != target.shape[3]:
            raise ValueError("KV 形状与已核对的 SDK 布局不符.")
        length = raw.shape[2]
        if length > target.shape[2]:
            raise ValueError("KV 输出超过缓存容量.")
        if append:
            target[:, :, :-length, :] = target[:, :, length:, :].copy()
        else:
            target.fill(item["zero_point"])
        if source["scale"] == item["scale"] and source["zero_point"] == item["zero_point"]:
            converted = raw
        else:
            values = (raw.astype(np.float32) - source["zero_point"]) * source["scale"]
            converted = np.clip(np.rint(values / item["scale"] + item["zero_point"]),
                                -32768, 32767).astype("<i2")
        target[:, :, -length:, :] = converted


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
        self.input_storage = list(self.inputs)
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
                        item = contract["inputs"][index] if output == 0 else None
                        # 已实测 Decode 单行 mask 的末尾按 16 字节填充,其余布局仍严格检查.
                        if item is None or item["name"] != "mask" or \
                                np.prod(buffer.shape[:-1]) != 1 or \
                                actual.value != (buffer.nbytes + 15) // 16 * 16:
                            raise ValueError(f"DLA 缓冲区大小不符: output={output},index={index},"
                                             f"logical={buffer.nbytes},actual={actual.value}")
                        storage = np.full(actual.value // 2, -32768, dtype="<i2")
                        view = storage[:buffer.size].reshape(buffer.shape)
                        np.copyto(view, buffer)
                        self.input_storage[index] = storage
                        self.inputs[index] = view
                        print(f"[Runtime mask padding] {buffer.nbytes} -> {actual.value} 字节",
                              flush=True)
        except Exception:
            self.close()
            raise

    def run(self):
        """同步执行一次真实硬件推理并返回原始输出."""
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        inputs = (pointer * len(self.input_storage))(*(b.ctypes.data for b in self.input_storage))
        outputs = (pointer * len(self.outputs))(*(b.ctypes.data for b in self.outputs))
        input_sizes = (size * len(self.input_storage))(*(b.nbytes for b in self.input_storage))
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
    if report.get("evaluation_scope") == "subset" and not is_test():
        raise ValueError("快速语料必须使用 -test,不能作为正式评测输入.")
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
    if args.diagnostic_blocks:
        token_blocks = token_blocks[:args.diagnostic_blocks]
    if is_test():
        record_selection(f"{index}:{row.tolist()}" for index, row in enumerate(token_blocks))
    args.output.mkdir(parents=True, exist_ok=True)
    model = HardwareModel(args.library, model_path, contract)
    sdk_inputs = np.load(inputs_path)
    for index, item in enumerate(contract["inputs"]):
        if item["name"] in ("mask", "pos_emb"):
            np.copyto(model.inputs[index], sdk_inputs[item["name"]])
    output_spec = contract["outputs"][0]
    rows, nll_values = [], []
    started = time.perf_counter()
    decode_model = None
    try:
        if args.decode_steps:
            decode_folder = args.quality_data / "decode"
            decode_contract = json.loads((decode_folder / "board_input_contract.json").read_text())
            decode_inputs_path = decode_folder / "board_inputs.npz"
            if sha256(decode_inputs_path) != decode_contract["board_inputs_sha256"]:
                raise ValueError("Decode SDK 输入哈希不符.")
            decode_arrays = np.load(decode_inputs_path)
            decode_path = args.package / f"{manifest['context']}c/decode.dla"
            entry = next(item for item in manifest["files"] if item["name"] ==
                         decode_path.relative_to(args.package).as_posix())
            if sha256(decode_path) != entry["sha256"]:
                raise ValueError("Decode DLA 哈希不符.")
            sequences = report["decode_input_ids"][:len(token_blocks)]
            if any(len(sequence) != 128 + args.decode_steps for sequence in sequences):
                raise ValueError("Decode 真实 Token 长度不符.")
            decode_model = HardwareModel(args.library, decode_path, decode_contract)
        for index, tokens in enumerate(token_blocks):
            np.copyto(model.inputs[0], embedding[tokens][None, :, :])
            tick = time.perf_counter()
            all_raw = model.run()[0][0]
            raw = all_raw[:-1]
            inference_seconds = time.perf_counter() - tick
            logits = (raw.astype(np.float64) - output_spec["zero_point"]) * output_spec["scale"]
            if args.diagnostic_blocks:
                np.save(args.output / f"npu_logits_{index:03d}.npy",
                        logits.astype(np.float32))
                last = (all_raw[-1:].astype(np.float32) - output_spec["zero_point"]) * output_spec["scale"]
                np.save(args.output / f"npu_prefill_last_logits_{index:03d}.npy", last)
            maximum = logits.max(axis=-1)
            normalizer = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=-1))
            nll = normalizer - logits[np.arange(127), tokens[1:]]
            nll_values.extend(nll.tolist())
            rows.append({"block": index, "inference_seconds": inference_seconds})
            if args.decode_steps:
                sequence = sequences[index]
                if sequence[:128] != tokens.tolist():
                    raise ValueError("Prefill 与 Decode 上下文不符.")
                transfer_cache(contract, model.outputs, decode_contract, decode_model.inputs)
                decode_logits = []
                decode_tick = time.perf_counter()
                for step, token in enumerate(sequence[128:]):
                    np.copyto(decode_model.inputs[0], embedding[token][None, None, :])
                    for offset, item in enumerate(decode_contract["inputs"]):
                        if item["name"] in ("mask", "pos_emb"):
                            np.copyto(decode_model.inputs[offset], decode_arrays[item["name"]][step])
                    decode_outputs = decode_model.run()
                    spec = decode_contract["outputs"][0]
                    values = (decode_outputs[0][0, 0].astype(np.float32) - spec["zero_point"]) * spec["scale"]
                    decode_logits.append(values)
                    if step + 1 < args.decode_steps:
                        transfer_cache(decode_contract, decode_outputs, decode_contract,
                                       decode_model.inputs, append=True)
                np.save(args.output / f"npu_decode_logits_{index:03d}.npy", np.stack(decode_logits))
                rows[-1]["decode_inference_seconds"] = time.perf_counter() - decode_tick
            print(f"[MDLA PPL] {index + 1}/{len(token_blocks)}", flush=True)
    finally:
        if decode_model is not None:
            decode_model.close()
        model.close()
    result = {key: report[key] for key in ("protocol", "model_name", "corpus_sha256",
              "selected_text_sha256", "blocks", "block_tokens", "scored_tokens",
              "excluded_tail_tokens", "input_ids", "reset_context_each_block")}
    result.update(backend="neuron_mdla_hardware", scope="board_fixed_text_subset_not_full_benchmark",
                  graph_mode="prefill_128_teacher_forcing",
                  perplexity=float(np.exp(np.mean(nll_values))), nll=nll_values,
                  elapsed_seconds=time.perf_counter() - started, block_timings=rows,
                  dla_sha256=sha256(model_path), embedding_sha256=sha256(embedding_path),
                  bridge_sha256=sha256(args.library), cpu_fallback=False)
    if args.diagnostic_blocks:
        result.update(scope="logits_diagnostic_prefix_not_full_accuracy",
                      blocks=len(token_blocks), scored_tokens=len(nll_values),
                      input_ids=token_blocks.reshape(-1).tolist(),
                      diagnostic_blocks=len(token_blocks))
    if args.decode_steps:
        result.update(decode_steps=args.decode_steps, decode_input_ids=sequences,
                      decode_dla_sha256=sha256(decode_path),
                      decode_tflite_sha256=decode_contract["tflite_sha256"],
                      decode_graph_mode="teacher_forcing_with_backend_own_prefill_cache")
    args.output.mkdir(parents=True, exist_ok=True)
    result.update(status=result_status(), **scope_metadata())
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
    parser.add_argument("--diagnostic-blocks", type=int, default=0,
                        help="仅执行前 N 个块并保存逐位置硬件 logits.")
    parser.add_argument("--decode-steps", type=int, default=0)
    add_test_argument(parser)
    args = parser.parse_args()
    if is_test():
        args.output = args.output / "test"
        args.diagnostic_blocks = 4
    if args.diagnostic_blocks < 0:
        parser.error("诊断块数不能为负数.")
    if not 0 <= args.decode_steps <= 32 or (args.decode_steps and not args.diagnostic_blocks):
        parser.error("Decode 步数须在 [0,32],且需要诊断模式.")
    evaluate(args)


if __name__ == "__main__":
    main()
