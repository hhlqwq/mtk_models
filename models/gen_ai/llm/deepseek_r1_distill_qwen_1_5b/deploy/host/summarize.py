"""汇总同协议参考与真实板端结果,回收可读中英文输出."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def summarize(args):
    """检查样本覆盖并计算 PPL、差异及性能,不将样例标为正式精度."""
    reference = json.loads((args.models / "reference.json").read_text())
    samples = json.loads((args.models / "samples.json").read_text())
    predictions = json.loads((args.results / "predictions.json").read_text())
    summary = json.loads((args.results / "board_summary.json").read_text())
    expected = {sample["id"] for sample in samples["samples"]}
    if len(predictions) != len(expected) or {row["id"] for row in predictions} != expected:
        raise ValueError("板端结果缺失或重复.")
    onnx = {row["id"]: row for row in reference["onnx"]}
    pytorch = {row["id"]: row for row in reference["pytorch"]}
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    output = args.results / "examples" / "output"
    output.mkdir(parents=True, exist_ok=True)
    all_nll = {"pytorch": [], "onnx": [], "npu": []}
    agreement = []
    errors = []
    rows = []
    for prediction in predictions:
        sample_id = prediction["id"]
        text = tokenizer.decode(prediction["tokens"], skip_special_tokens=False)
        sample = next(row for row in samples["samples"] if row["id"] == sample_id)
        if prediction["nll"]:
            expected_count = len(sample["target_ids"])
            if len(prediction["nll"]) != expected_count:
                raise ValueError(f"教师强制 Token 数量不匹配: {sample_id}")
            for backend in ("pytorch", "onnx"):
                all_nll[backend].extend((pytorch if backend == "pytorch" else onnx)[sample_id]["nll"])
            all_nll["npu"].extend(prediction["nll"])
            agreement.extend(np.array(prediction["argmax"]) == onnx[sample_id]["argmax"])
            errors.extend(np.abs(np.array(prediction["nll"]) - onnx[sample_id]["nll"]).tolist())
        else:
            (output / f"{sample_id}.txt").write_text(text, encoding="utf-8")
            (output / f"{sample_id}_pytorch.txt").write_text(
                tokenizer.decode(pytorch[sample_id]["tokens"], skip_special_tokens=False),
                encoding="utf-8")
        steps = max(len(prediction["tokens"]) - 1, 0)
        rows.append({**prediction, "text": text,
                     "exact_token_match_pytorch": prediction["tokens"] ==
                     pytorch[sample_id]["tokens"] if not prediction["nll"] else None,
                     "prompt": sample.get("prompt"), "reference_text": sample.get("text"),
                     "input_ids": sample["input_ids"],
                     "reference_nll": {"pytorch": pytorch[sample_id]["nll"],
                                       "onnx": onnx[sample_id]["nll"]}
                     if prediction["nll"] else None,
                     "truncated": not prediction["nll"] and not prediction["reached_eos"],
                     "has_final_answer": not prediction["nll"] and "</think>" in text,
                     "decode_tokens_per_s": steps * 1000 / prediction["decode_ms"]
                     if steps else None,
                     "prefill_tokens_per_s": prediction["prefill_tokens"] * 1000 /
                     prediction["ttft_ms"]})
    if not all_nll["npu"] or not np.isfinite(all_nll["npu"]).all():
        raise ValueError("缺少有效质量评测 Token.")
    summary.update({"model": "DeepSeek-R1-Distill-Qwen-1.5B",
                    "source_revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
                    "ppl": {key: math.exp(float(np.mean(value)))
                            for key, value in all_nll.items()},
                    "quality_tokens": len(all_nll["npu"]),
                    "onnx_npu_argmax_agreement": float(np.mean(agreement)),
                    "onnx_npu_max_token_nll_error": max(errors),
                    "export_max_token_nll_error": reference["export_max_token_nll_error"],
                    "samples": rows,
                    "formal_accuracy_verified": False,
                    "max_new_tokens": samples["max_new_tokens"],
                    "prefill_mode": "sequential_single_token",
                    "cpu_work": "embedding_lookup, RoPE, KV management, sampling, metrics",
                    "npu_work": "attention, normalization, MLP, vocabulary projection"})
    artifacts = json.loads((args.models / "artifact_manifest.json").read_text())
    summary["deployment_bytes"] = sum(row["bytes"] for row in artifacts["files"])
    summary["dla_count"] = sum(row["name"].endswith(".dla") for row in artifacts["files"])
    summary["context"] = json.loads(
        (args.models / "export_manifest.json").read_text())["context"]
    calls = sum(row["prefill_tokens"] + len(row["tokens"]) - 1 for row in predictions)
    summary["mean_npu_ms_per_token"] = sum(row["npu_ms"] for row in predictions) / calls
    # 数值门槛只用于排查导出或布局问题,不能代替产品质量标准.
    summary["numerical_check_passed"] = max(errors) <= 0.25
    summary["status"] = ("board_smoke_verified" if summary["numerical_check_passed"]
                         else "board_numerical_check_failed")
    (args.results / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in
                      ("ppl", "quality_tokens", "onnx_npu_argmax_agreement",
                       "onnx_npu_max_token_nll_error", "peak_rss_mib")},
                     ensure_ascii=False, indent=2))
    if not summary["numerical_check_passed"]:
        raise RuntimeError("板端数值差异超过排查门槛,结果已保存,不得标记交付完成.")


def parse_args():
    """解析官方 Tokenizer、参考产物和已回收板端结果目录."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    summarize(parse_args())
