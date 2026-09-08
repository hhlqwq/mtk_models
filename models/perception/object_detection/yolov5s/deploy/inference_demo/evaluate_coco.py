"""在开发板使用官方 pycocotools 计算 COCO bbox 指标."""

import argparse
import contextlib
import io
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


METRIC_NAMES = [
    "AP_50_95",
    "AP_50",
    "AP_75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def load_processed_ids(path: Path) -> set[int]:
    """读取 C++ 评测程序写出的已完成图片集合."""
    return {
        int(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def evaluate(args: argparse.Namespace) -> None:
    """验证覆盖范围,执行 COCOeval 并保存指标和原始摘要."""
    annotation = COCO(str(args.annotations))
    expected_ids = set(annotation.getImgIds())
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

    prediction = annotation.loadRes(str(args.predictions))
    evaluator = COCOeval(annotation, prediction, "bbox")
    evaluator.params.imgIds = sorted(expected_ids)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    summary_text = output.getvalue()
    print(summary_text, end="")
    args.summary_log.write_text(summary_text, encoding="utf-8")

    metrics = {
        "images": len(expected_ids),
        "prediction_records": len(predictions),
        "metrics": {
            name: float(value)
            for name, value in zip(METRIC_NAMES, evaluator.stats)
        },
    }
    args.metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] COCO 指标: {args.metrics}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--processed-ids", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--summary-log", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
