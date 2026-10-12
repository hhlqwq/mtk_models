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
    spec = contract["outputs"][0]
    args.output.mkdir(parents=True, exist_ok=False)
    for index, tokens in enumerate(token_blocks):
        inputs["input_embeds"] = embedding[tokens][None, :, :]
        outputs = model.run(inputs, output_names=[spec["name"]],
                            quantize_input=False, dequantize_output=False)
        raw = outputs[spec["name"]] if isinstance(outputs, dict) else outputs[0]
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        values = (np.asarray(raw)[0, :-1].astype(np.float32) - spec["zero_point"]) * spec["scale"]
        np.save(args.output / f"tflite_logits_{index:03d}.npy", values)
        print(f"[TFLite logits] {index + 1}/{args.blocks}", flush=True)
    report = {"scope": "logits_diagnostic_prefix_not_full_accuracy",
              "input_ids": token_blocks.reshape(-1).tolist(), "blocks": args.blocks,
              "tflite_sha256": sha256(args.graph), "embedding_sha256": sha256(args.embedding),
              "board_inputs_sha256": sha256(inputs_path), "raw_int16_inputs": True}
    (args.output / "tflite_quality.json").write_text(json.dumps(report, indent=2) + "\n")


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
    blocks = len(ids) // 128
    pairs = {"pytorch_tflite": [], "pytorch_npu": [], "tflite_npu": []}
    files = []
    for index in range(blocks):
        values = {}
        for name, path in folders.items():
            source = path / f"{name}_logits_{index:03d}.npy"
            values[name] = np.load(source)
            files.append({"backend": name, "block": index, "sha256": sha256(source)})
        for name in pairs:
            left, right = name.split("_")
            pairs[name].append(compare_pair(values[left], values[right]))
        print(f"[logits 对比] {index + 1}/{blocks}", flush=True)
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
    result = {"scope": "prefill_logits_diagnostic_prefix_not_full_accuracy",
              "blocks": blocks, "positions": blocks * 127, "input_ids": ids,
              "decode_accuracy_verified": False, "quality_qualified": False,
              "threshold": "not_assigned_no_pass_claim", "summary": summary,
              "details": pairs, "files": files}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logits_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    """解析静态图导出与三后端配对比较参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tflite", "compare"))
    for name in ("quality-data", "graph", "embedding", "reference", "tflite", "npu", "output"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--blocks", type=int, default=4)
    args = parser.parse_args()
    required = ("quality_data", "graph", "embedding", "output") if args.mode == "tflite" \
        else ("reference", "tflite", "npu", "output")
    if args.blocks < 1 or any(getattr(args, name) is None for name in required):
        parser.error("缺少必需路径或块数无效.")
    export_tflite(args) if args.mode == "tflite" else summarize(args)


if __name__ == "__main__":
    main()
