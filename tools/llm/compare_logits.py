"""以相同 INT16 输入执行主机静态图,比较三个后端的逐位置 logits."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    """分块计算真实资源哈希."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_tflite(args):
    """用官方执行器运行原始 INT16 输入,保持与硬件输入一致."""
    from mtk_converter.python.tools.tflite_executor import TFLiteExecutor
    from evaluate_board_quality import transfer_cache

    contract = json.loads((args.quality_data / "board_input_contract.json").read_text())
    protocol = json.loads((args.quality_data / "pytorch_quality.json").read_text())
    if sha256(args.graph) != contract["tflite_sha256"]:
        raise ValueError("TFLite 图与已验证契约不同.")
    inputs_path = args.quality_data / "board_inputs.npz"
    if sha256(inputs_path) != contract["board_inputs_sha256"]:
        raise ValueError("SDK mask/RoPE 资源哈希不符.")
    if args.blocks > protocol["blocks"]:
        raise ValueError("诊断块数超过原协议.")
    embedding = np.memmap(args.embedding, dtype="<i2", mode="r").reshape(
        -1, contract["inputs"][0]["shape"][-1])
    token_blocks = np.array(protocol["input_ids"]).reshape(-1, 128)[:args.blocks]
    arrays = np.load(inputs_path)
    inputs = {item["name"]: np.full(item["shape"], item["zero_point"], dtype="<i2")
              for item in contract["inputs"]}
    for name in ("mask", "pos_emb"):
        inputs[name] = arrays[name]
    print("[TFLite logits] 加载已编译的对应静态图.", flush=True)
    model = TFLiteExecutor(str(args.graph), cache_dequantized_weight=True)
    decode_model = None
    if args.decode_steps:
        decode_contract = json.loads((args.quality_data / "decode/board_input_contract.json").read_text())
        if sha256(args.decode_graph) != decode_contract["tflite_sha256"]:
            raise ValueError("Decode 图与契约哈希不同.")
        decode_inputs_path = args.quality_data / "decode/board_inputs.npz"
        if sha256(decode_inputs_path) != decode_contract["board_inputs_sha256"]:
            raise ValueError("Decode SDK 输入资源不符.")
        decode_arrays = np.load(decode_inputs_path)
        sequences = protocol["decode_input_ids"][:args.blocks]
        decode_inputs = [np.full(item["shape"], item["zero_point"], dtype="<i2")
                         for item in decode_contract["inputs"]]
        decode_model = TFLiteExecutor(str(args.decode_graph), cache_dequantized_weight=True)
    spec = contract["outputs"][0]
    args.output.mkdir(parents=True, exist_ok=False)
    for index, tokens in enumerate(token_blocks):
        inputs["input_embeds"] = embedding[tokens][None, :, :]
        names = [item["name"] for item in contract["outputs"]] if args.decode_steps else [spec["name"]]
        outputs = model.run(inputs, output_names=names,
                            quantize_input=False, dequantize_output=False)
        raw = outputs[spec["name"]] if isinstance(outputs, dict) else outputs[0]
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        values = (np.asarray(raw)[0, :-1].astype(np.float32) - spec["zero_point"]) * spec["scale"]
        np.save(args.output / f"tflite_logits_{index:03d}.npy", values)
        last = (np.asarray(raw)[0, -1:].astype(np.float32) - spec["zero_point"]) * spec["scale"]
        np.save(args.output / f"tflite_prefill_last_logits_{index:03d}.npy", last)
        if args.decode_steps:
            raw_outputs = raw_output_list(outputs, contract)
            transfer_cache(contract, raw_outputs, decode_contract, decode_inputs)
            values = []
            for step, token in enumerate(sequences[index][128:]):
                decode_inputs[0] = embedding[token][None, None, :].copy()
                for offset, item in enumerate(decode_contract["inputs"]):
                    if item["name"] in ("mask", "pos_emb"):
                        decode_inputs[offset] = decode_arrays[item["name"]][step]
                feed = {item["name"]: value for item, value in
                        zip(decode_contract["inputs"], decode_inputs)}
                outputs = decode_model.run(feed, quantize_input=False, dequantize_output=False)
                raw_outputs = raw_output_list(outputs, decode_contract)
                output_spec = decode_contract["outputs"][0]
                logits = (raw_outputs[0][0, 0].astype(np.float32) - output_spec["zero_point"]) * output_spec["scale"]
                values.append(logits)
                if step + 1 < args.decode_steps:
                    transfer_cache(decode_contract, raw_outputs, decode_contract,
                                   decode_inputs, append=True)
            np.save(args.output / f"tflite_decode_logits_{index:03d}.npy", np.stack(values))
        print(f"[TFLite logits] {index + 1}/{args.blocks}", flush=True)
    report = {"scope": "logits_diagnostic_prefix_not_full_accuracy",
              "input_ids": token_blocks.reshape(-1).tolist(), "blocks": args.blocks,
              "tflite_sha256": sha256(args.graph), "embedding_sha256": sha256(args.embedding),
              "board_inputs_sha256": sha256(inputs_path), "raw_int16_inputs": True}
    if args.decode_steps:
        report.update(decode_steps=args.decode_steps, decode_input_ids=sequences,
                      decode_tflite_sha256=sha256(args.decode_graph),
                      decode_graph_mode="teacher_forcing_with_backend_own_prefill_cache")
    (args.output / "tflite_quality.json").write_text(json.dumps(report, indent=2) + "\n")


def raw_output_list(outputs, contract):
    """按实际图契约获取全部原始 INT16 输出."""
    values = [outputs[item["name"]] for item in contract["outputs"]] \
        if isinstance(outputs, dict) else list(outputs)
    arrays = []
    if len(values) != len(contract["outputs"]):
        raise ValueError("TFLite 输出数量不符.")
    for value, item in zip(values, contract["outputs"]):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.shape != tuple(item["shape"]) or array.dtype != np.int16:
            raise ValueError(f"原始输出接口不符: {item['name']}")
        arrays.append(array)
    return arrays


def compare_pair(reference, candidate):
    """计算逐位置余弦、误差、Top1 和概率 KL,不据单一指标宣布验收通过."""
    if reference.shape != candidate.shape or not np.isfinite(reference).all() or \
            not np.isfinite(candidate).all():
        raise ValueError("输出形状不符或包含非有限数值.")
    left, right = reference.astype(np.float64), candidate.astype(np.float64)
    norms = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    if np.any(norms == 0):
        raise ValueError("零向量无法计算余弦相似度.")
    cosine = (left * right).sum(axis=-1) / norms
    centered_left = left - left.mean(axis=-1, keepdims=True)
    centered_right = right - right.mean(axis=-1, keepdims=True)
    centered_cosine = (centered_left * centered_right).sum(axis=-1) / (
        np.linalg.norm(centered_left, axis=-1) * np.linalg.norm(centered_right, axis=-1))
    difference = right - left
    left -= left.max(axis=-1, keepdims=True)
    right -= right.max(axis=-1, keepdims=True)
    log_left = left - np.log(np.exp(left).sum(axis=-1, keepdims=True))
    log_right = right - np.log(np.exp(right).sum(axis=-1, keepdims=True))
    return {"cosine": cosine.tolist(), "centered_cosine": centered_cosine.tolist(),
            "mse": np.mean(difference ** 2, axis=-1).tolist(),
            "mae": np.mean(np.abs(difference), axis=-1).tolist(),
            "max_abs_error": np.max(np.abs(difference), axis=-1).tolist(),
            "top1_equal": (reference.argmax(axis=-1) == candidate.argmax(axis=-1)).tolist(),
            "kl_reference_candidate": (np.exp(log_left) * (log_left - log_right)).sum(axis=-1).tolist()}


def summarize_pairs(pairs):
    """汇总逐位置误差,可复用到完整 Prefill 和最后一个输出位置."""
    summary = {}
    for name, rows in pairs.items():
        merged = {key: np.concatenate([np.array(row[key]) for row in rows]) for key in rows[0]}
        summary[name] = {"cosine_mean": float(merged["cosine"].mean()),
                         "cosine_min": float(merged["cosine"].min()),
                         "centered_cosine_mean": float(merged["centered_cosine"].mean()),
                         "rmse": float(np.sqrt(merged["mse"].mean())),
                         "mae": float(merged["mae"].mean()),
                         "max_abs_error": float(merged["max_abs_error"].max()),
                         "top1_agreement": float(merged["top1_equal"].mean()),
                         "kl_mean": float(merged["kl_reference_candidate"].mean())}
    return summary


def summarize(args):
    """核对相同 Token 与文件哈希,生成三个后端的配对诊断报告."""
    folders = {"pytorch": args.reference, "tflite": args.tflite, "npu": args.npu}
    reports = {name: json.loads((path / f"{name}_quality.json").read_text())
               for name, path in folders.items()}
    ids = reports["pytorch"]["input_ids"]
    if any(report["input_ids"] != ids for report in reports.values()):
        raise ValueError("三个后端的诊断输入 Token 不相同.")
    if reports["tflite"]["embedding_sha256"] != reports["npu"]["embedding_sha256"]:
        raise ValueError("主机与板端 embedding 不相同.")
    if args.decode_steps:
        sequences = reports["pytorch"]["decode_input_ids"]
        if any(report.get("decode_input_ids") != sequences for report in reports.values()):
            raise ValueError("三个后端的 Decode Token 不相同.")
        if reports["tflite"]["decode_tflite_sha256"] != reports["npu"]["decode_tflite_sha256"]:
            raise ValueError("Decode 主机图与板端契约不相同.")
    blocks = len(ids) // 128
    pairs = {"pytorch_tflite": [], "pytorch_npu": [], "tflite_npu": []}
    files = []
    for index in range(blocks):
        values = {}
        for name, path in folders.items():
            suffix = "decode_logits" if args.decode_steps else "logits"
            source = path / f"{name}_{suffix}_{index:03d}.npy"
            values[name] = np.load(source)
            files.append({"backend": name, "block": index, "sha256": sha256(source)})
        for name in pairs:
            left, right = name.split("_")
            pairs[name].append(compare_pair(values[left], values[right]))
        print(f"[logits 对比] {index + 1}/{blocks}", flush=True)
    summary = summarize_pairs(pairs)
    result = {"scope": "prefill_logits_diagnostic_prefix_not_full_accuracy",
              "blocks": blocks, "positions": blocks * 127, "input_ids": ids,
              "decode_accuracy_verified": False, "quality_qualified": False,
              "status": "diagnostic_evaluated_quality_not_qualified",
              "threshold": "not_assigned_no_pass_claim", "summary": summary,
              "details": pairs, "files": files}
    if not args.decode_steps:
        last_paths = {name: [path / f"{name}_prefill_last_logits_{index:03d}.npy"
                            for index in range(blocks)] for name, path in folders.items()}
        available = [path.exists() for paths in last_paths.values() for path in paths]
        if any(available) and not all(available):
            raise ValueError("Prefill 最后一行的三个后端证据不完整.")
        if all(available):
            last_pairs = {name: [] for name in pairs}
            for index in range(blocks):
                values = {name: np.load(paths[index]) for name, paths in last_paths.items()}
                for name in last_pairs:
                    left, right = name.split("_")
                    last_pairs[name].append(compare_pair(values[left], values[right]))
                for name, paths in last_paths.items():
                    files.append({"backend": name, "block": index,
                                  "stage": "prefill_last", "sha256": sha256(paths[index])})
            complete = {name: pairs[name] + last_pairs[name] for name in pairs}
            result.update(prefill_last_token_summary=summarize_pairs(last_pairs),
                          prefill_complete_output_summary=summarize_pairs(complete),
                          prefill_total_output_positions=blocks * 128,
                          prefill_last_positions=blocks)
    if args.decode_steps:
        result.update(scope="decode_teacher_forcing_diagnostic_not_full_accuracy",
                      positions=blocks * args.decode_steps, decode_input_ids=sequences,
                      decode_steps=args.decode_steps, decode_accuracy_evaluated=True)
    args.output.mkdir(parents=True, exist_ok=True)
    filename = "decode_logits_comparison.json" if args.decode_steps else "logits_comparison.json"
    (args.output / filename).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    """解析静态图导出与三后端配对比较参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tflite", "compare"))
    for name in ("quality-data", "graph", "decode-graph", "embedding", "reference", "tflite", "npu", "output"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=0)
    args = parser.parse_args()
    required = ("quality_data", "graph", "embedding", "output") if args.mode == "tflite" \
        else ("reference", "tflite", "npu", "output")
    if args.blocks < 1 or any(getattr(args, name) is None for name in required):
        parser.error("缺少必需路径或块数无效.")
    if not 0 <= args.decode_steps <= 32 or (args.mode == "tflite" and
                                         args.decode_steps and args.decode_graph is None):
        parser.error("Decode 步数须在 [0,32],TFLite 导出需要 --decode-graph.")
    export_tflite(args) if args.mode == "tflite" else summarize(args)


if __name__ == "__main__":
    main()
