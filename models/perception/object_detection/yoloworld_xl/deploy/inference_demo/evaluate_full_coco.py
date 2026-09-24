"""仅从 C++ Neuron EP 预测计算 COCO val2017 全量 bbox mAP."""

import argparse
import contextlib
import io
import json
import math
import statistics
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


def parse_args() -> argparse.Namespace:
    """解析指标阶段的预测、标注、profiling 与报告路径."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def percentile(values: list[float], percent: float) -> float:
    """对耗时样本计算线性插值百分位数."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def read_profile(path: Path) -> dict[str, int]:
    """统计实际执行的 ORT 节点并要求存在 Neuron EP 事件."""
    events = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for event in events:
        provider = event.get("args", {}).get("provider")
        if provider:
            counts[provider] = counts.get(provider, 0) + 1
    if not any("Neuron" in name and count > 0 for name, count in counts.items()):
        raise ValueError("profiling 中缺少 Neuron EP 节点执行证据.")
    return counts


def evaluate(args: argparse.Namespace) -> None:
    """核对 5000 张 C++ 输出、计算 bbox AP 并生成小体积报告."""
    provider_counts = read_profile(args.profile)
    annotation = COCO(str(args.annotations))
    expected = set(annotation.getImgIds())
    categories = {item["name"]: item["id"]
                  for item in annotation.dataset["categories"]}
    if set(COCO_CLASSES) != set(categories):
        raise ValueError("模型 COCO 类名与标注不一致.")
    seen = set()
    predictions = []
    timings = []
    with args.results.open(encoding="utf-8") as source:
        for index, line in enumerate(source, start=1):
            record = json.loads(line)
            image_id = int(Path(record["image"]).stem)
            if image_id in seen or image_id not in expected:
                raise ValueError(f"重复或未知图片 ID: {image_id}.")
            seen.add(image_id)
            elapsed = float(record["inference_ms"])
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError(f"推理耗时非法: {image_id}.")
            timings.append(elapsed)
            for item in record["detections"]:
                category = int(item["class_id"])
                if not 0 <= category < 80:
                    raise ValueError(f"检测类别越界: {category}.")
                left, top, right, bottom = map(float, item["bbox_xyxy"])
                score = float(item["score"])
                if (not all(map(math.isfinite, (left, top, right, bottom,
                                                 score))) or
                        right < left or bottom < top):
                    raise ValueError(f"检测框无效: {image_id}.")
                predictions.append({
                    "image_id": image_id,
                    "category_id": categories[COCO_CLASSES[category]],
                    "bbox": [left, top, right - left, bottom - top],
                    "score": score,
                })
            if index % 500 == 0:
                print(f"[METRIC] COCO {index}/5000 张.", flush=True)
    if seen != expected or len(timings) != 5000:
        raise ValueError(f"COCO 覆盖不完整: {len(seen)}/5000.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir.parent / "coco_predictions.json"
    predictions_path.write_text(
        json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    result = annotation.loadRes(str(predictions_path))
    evaluator = COCOeval(annotation, result, "bbox")
    evaluator.params.imgIds = sorted(expected)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    (args.output_dir / "coco_summary.log").write_text(
        log.getvalue(), encoding="utf-8")
    summary = {
        "status": "complete",
        "model": "yoloworld_xl",
        "run_id": args.run_id,
        "dataset": "coco_val2017",
        "images": 5000,
        "prediction_records": len(predictions),
        "accuracy": {
            "AP_50_95": float(evaluator.stats[0]),
            "AP_50": float(evaluator.stats[1]),
            "AP_75": float(evaluator.stats[2]),
        },
        "timing_scope": "C++ ONNX Runtime Neuron EP session.Run only",
        "timing": {
            "count": len(timings),
            "mean_ms": statistics.fmean(timings),
            "p50_ms": percentile(timings, 50),
            "p90_ms": percentile(timings, 90),
            "p95_ms": percentile(timings, 95),
        },
        "profile": {
            "scope": "three warmup runs in the same persistent Neuron EP session",
            "provider_node_events": provider_counts,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[OK] YOLO-World XL 全量 COCO mAP: {args.output_dir}.")


if __name__ == "__main__":
    evaluate(parse_args())
