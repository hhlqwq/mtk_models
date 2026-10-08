"""在开发板汇总 NPU 平均耗时和 COCO mAP 量化精度变化."""

import argparse
import contextlib
import io
import json
import math
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_processed_ids(path: Path) -> set[int]:
    """读取 C++ 评测程序写出的已完成图片集合."""
    return {
        int(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def evaluate(args: argparse.Namespace) -> None:
    """校验全量覆盖,计算 mAP 并输出耗时和精度变化摘要."""
    print("[COCO] 加载标注与预测,校验图片覆盖范围.", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        annotation = COCO(str(args.annotations))
    expected_ids = set(annotation.getImgIds())
    if len(expected_ids) != 5000:
        raise ValueError("COCO val2017 全量标注必须包含 5000 张图片.")
    processed_ids = load_processed_ids(args.processed_ids)
    if processed_ids != expected_ids:
        missing = sorted(expected_ids - processed_ids)
        unexpected = sorted(processed_ids - expected_ids)
        raise RuntimeError(
            "评测覆盖范围不完整: "
            f"expected={len(expected_ids)}, processed={len(processed_ids)}, "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}")

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    prediction_ids = {int(item["image_id"]) for item in predictions}
    if not prediction_ids.issubset(expected_ids):
        unexpected = sorted(prediction_ids - expected_ids)
        raise RuntimeError(f"预测包含未知 image_id: {unexpected[:5]}")

    with contextlib.redirect_stdout(io.StringIO()):
        prediction = annotation.loadRes(str(args.predictions))
    evaluator = COCOeval(annotation, prediction, "bbox")
    evaluator.params.imgIds = sorted(expected_ids)
    print("[COCO] 计算逐图精度,请等待.", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
    print("[COCO] 汇总 mAP@0.5:0.95.", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.accumulate()
        # 官方 summarize 会计算 stats,仅隐藏其余 11 项的打印.
        evaluator.summarize()
    int8_map = float(evaluator.stats[0])
    if not math.isfinite(int8_map) or not 0.0 <= int8_map <= 1.0:
        raise ValueError("COCO mAP 无效,无法汇总精度.")

    timings = json.loads(args.timings.read_text(encoding="utf-8"))
    if timings["processed_images"] != len(expected_ids):
        raise ValueError("耗时报告与精度报告的图片数不一致.")
    npu_mean = float(timings["npu_ms"]["mean"])
    if not math.isfinite(npu_mean) or npu_mean <= 0.0:
        raise ValueError("NPU 平均耗时无效.")
    peak_rss = timings.get("peak_rss_kb")
    if peak_rss is not None and (not math.isfinite(float(peak_rss)) or float(peak_rss) <= 0):
        raise ValueError("峰值 RSS 无效.")
    peak_rss_mib = None if peak_rss is None else float(peak_rss) / 1024.0
    fp32_map = args.fp32_map
    if fp32_map is not None and (
            not math.isfinite(fp32_map) or not 0.0 <= fp32_map <= 1.0):
        raise ValueError("FP32 mAP 必须是 0 到 1 之间的数值.")
    loss_pp = None if fp32_map is None else (fp32_map - int8_map) * 100.0
    summary_lines = [
        f"板端 NPU 平均推理耗时: {npu_mean:.3f} ms",
        f"INT8 mAP@0.5:0.95: {int8_map:.6f}",
    ]
    summary_lines.append("推理进程峰值 RSS: 未记录." if peak_rss_mib is None else
                         f"推理进程峰值 RSS: {peak_rss_mib:.3f} MiB")
    if fp32_map is None:
        summary_lines.append("量化精度下降: 未计算,请配置同协议 FP32_MAP 基准.")
    else:
        summary_lines.extend([
            f"FP32 mAP@0.5:0.95: {fp32_map:.6f}",
            f"精度下降 (FP32 - INT8): {loss_pp:.4f} 个百分点",
            f"FP32 基准来源: {args.fp32_source}",
        ])
    summary_text = "\n".join(summary_lines) + "\n"
    metrics = {
        "status": "complete",
        "model": "yolov5s",
        "run_id": args.run_id or args.metrics.parent.name,
        "dataset": "coco_val2017",
        "images": len(expected_ids),
        "npu_mean_ms": npu_mean,
        "peak_rss_mib": peak_rss_mib,
        "memory_scope": "板端 C++ 全量推理进程峰值 RSS",
        "int8_map_50_95": int8_map,
        "fp32_map_50_95": fp32_map,
        "accuracy_loss_percentage_points": loss_pp,
        "fp32_baseline_source": args.fp32_source if fp32_map is not None else None,
    }
    args.metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_text, end="", flush=True)


def parse_args() -> argparse.Namespace:
    """解析输入报告路径和同协议 FP32 精度基准."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--processed-ids", type=Path, required=True)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--fp32-map", type=float)
    parser.add_argument("--fp32-source", default="用户提供的同协议 FP32 基准")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--run-id")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
