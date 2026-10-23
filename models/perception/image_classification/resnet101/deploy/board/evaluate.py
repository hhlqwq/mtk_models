"""核对 ResNet101 全量预测并发布唯一精度、性能和 RSS 报告."""

import json
import math
import os
import platform
import statistics

from resnet101 import MODEL, PROTOCOL, WEIGHTS, label_signature, read_labels, validate_contract
from summary import publish
from tools.board import context
from tools.evaluation import is_test
from tools.files import read_json, sha256


def matched_reference(ctx, metadata: dict, labels_digest: str):
    """只接受相同 ONNX、标签映射和预处理的完整参考,缺失项如实记录."""
    path = ctx.root / "models/onnx_reference.json"
    if not path.is_file():
        return None, ["缺少同协议 FP32 ONNX 参考,精度差值未计算."]
    baseline = read_json(path)
    required = {
        "model": MODEL, "weights": WEIGHTS, "status": "complete",
        "evaluation_scope": "full", "protocol": PROTOCOL, "samples": 50000,
        "onnx_sha256": metadata["onnx_sha256"], "labels_sha256": labels_digest,
        "pillow": metadata["pillow"],
    }
    if any(baseline.get(key) != value for key, value in required.items()):
        raise ValueError("FP32 参考与本次模型、标签或评价协议不匹配,请重新准备参考.")
    accuracy = baseline["accuracy"]
    if not all(math.isfinite(accuracy[name]) and 0 <= accuracy[name] <= 1
               for name in ("top1", "top5")):
        raise ValueError("参考 Top-1/Top-5 无效.")
    return accuracy, []


def main() -> None:
    """评分不补跑推理,完整覆盖后才发布一个报告."""
    if is_test():
        raise ValueError("编译示例不得发布正式评价报告.")
    ctx = context(__file__)
    metadata = read_json(ctx.root / "models/quantization.json")
    validate_contract(metadata)
    run = read_json(ctx.work / "inference.json")
    labels = read_labels(ctx.dataset / "val_labels_0based.txt")
    labels_digest = label_signature(labels)
    required = {
        "model": MODEL, "run_id": ctx.run_id, "samples": 50000,
        "evaluation_scope": "full", "backend": "python_neuron_runtime_hw",
        "onnx_sha256": metadata["onnx_sha256"], "dla_sha256": metadata["dla_sha256"],
        "labels_sha256": labels_digest,
    }
    if any(run.get(key) != value for key, value in required.items()):
        raise ValueError("推理记录与本次模型、运行编号或完整评价范围不匹配.")
    if sha256(ctx.root / "models/model_int8.dla") != run["dla_sha256"]:
        raise ValueError("推理后 DLA 已发生变化.")
    baseline, unavailable = matched_reference(ctx, metadata, labels_digest)
    calibration_images = set(metadata["calibration"]["images"])
    overlap = set()
    top1 = top5 = 0
    holdout_top1 = holdout_top5 = 0
    npu_times, end_to_end_times = [], []
    with (ctx.work / "predictions.jsonl").open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= 50000:
                raise ValueError("预测数量超过 50000.")
            row = json.loads(line)
            name = f"ILSVRC2012_val_{index + 1:08d}.JPEG"
            predictions = row["top5"]
            if (row.get("index") != index or row.get("image") != name
                    or row.get("backend") != "python_neuron_runtime_hw"
                    or not isinstance(predictions, list) or len(predictions) != 5
                    or any(type(value) is not int or not 0 <= value < 1000 for value in predictions)
                    or len(set(predictions)) != 5):
                raise ValueError(f"预测记录不匹配: {name}.")
            npu, elapsed = row["npu_ms"], row["end_to_end_ms"]
            if not math.isfinite(npu) or npu <= 0 or not math.isfinite(elapsed) or elapsed < npu:
                raise ValueError(f"性能记录无效: {name}.")
            npu_times.append(npu)
            end_to_end_times.append(elapsed)
            correct1, correct5 = predictions[0] == labels[index], labels[index] in predictions
            top1 += correct1
            top5 += correct5
            if name in calibration_images:
                overlap.add(name)
            else:
                holdout_top1 += correct1
                holdout_top5 += correct5
            if (index + 1) % 5000 == 0:
                print(f"[评分] {MODEL} {index + 1}/50000.", flush=True)
    if len(npu_times) != 50000:
        raise ValueError(f"预测覆盖不完整: {len(npu_times)}/50000.")
    if type(run["peak_rss_kib"]) is not int or run["peak_rss_kib"] <= 0 or run["warmup_runs"] != 5:
        raise ValueError("峰值 RSS 或预热记录无效.")
    accuracy = {"top1": top1 / 50000, "top5": top5 / 50000}
    holdout_count = 50000 - len(overlap)
    report = {
        "schema_version": 1, "model": MODEL, "weights": WEIGHTS, "run_id": ctx.run_id,
        "status": "complete" if baseline else "partial", "evaluation_scope": "full",
        "protocol": PROTOCOL, "samples": 50000, "precision": metadata["precision"],
        "model_version": {"torchvision": metadata["torchvision"],
                          "converter": metadata["converter"],
                          "compiler_target": metadata["compiler_target"]},
        "accuracy": accuracy, "reference_accuracy": baseline,
        "accuracy_change_percentage_points": None if baseline is None else {
            name: (accuracy[name] - baseline[name]) * 100 for name in ("top1", "top5")},
        "calibration": {"samples": metadata["calibration"]["samples"],
                        "overlapping_validation_samples": len(overlap)},
        "holdout": {"samples": holdout_count,
                    "top1": holdout_top1 / holdout_count if holdout_count else None,
                    "top5": holdout_top5 / holdout_count if holdout_count else None,
                    "reference_accuracy": None},
        "performance": {
            "batch_size": 1, "warmup_runs": run["warmup_runs"], "measured_runs": 50000,
            "npu_mean_ms": statistics.fmean(npu_times),
            "npu_p95_ms": sorted(npu_times)[math.ceil(0.95 * 50000) - 1],
            "npu_timing_scope": "SDK 同步 inference 调用,不含 I/O 绑定与前后处理",
            "end_to_end_mean_ms": statistics.fmean(end_to_end_times),
            "end_to_end_timing_scope": "图片读取、预处理、量化、I/O 绑定、NPU 和 Top-5,不含模型加载及预测写盘",
            "peak_process_rss_mib": run["peak_rss_kib"] / 1024,
            "memory_scope": "Python 推理进程全生命周期 RSS 高水位,不代表全部驱动/NPU 内存",
        },
        "runtime": {"backend": "python_neuron_runtime_hw", "machine": platform.machine(),
                    "kernel": platform.release(), "pillow": run["pillow"],
                    "python": run["python"],
                    "calibration_pillow": metadata["pillow"],
                    "native_input_padded_bytes": run["input_padded_bytes"],
                    "native_output_padded_bytes": run["output_padded_bytes"]},
        "model_files": [{"file": "models/model_int8.dla",
                         "bytes": (ctx.root / "models/model_int8.dla").stat().st_size}],
        "unavailable_metrics": unavailable + ["holdout 的同子集 FP32 参考未单独计算."],
    }
    publish(ctx, report)


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    main()
