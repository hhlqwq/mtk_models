#!/usr/bin/env python3
"""比较 YOLO-World CPU 与 Neuron EP 检测结果."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析结果路径和一致性阈值."""
    parser = argparse.ArgumentParser(description="比较 CPU 与 Neuron 检测结果.")
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--neuron", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-score-delta", type=float, default=0.01)
    parser.add_argument("--max-bbox-delta", type=float, default=0.5)
    return parser.parse_args()


def load_report(path: Path) -> dict[str, object]:
    """读取板端结果 JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def compare_reports(
    cpu: dict[str, object],
    neuron: dict[str, object],
) -> dict[str, object]:
    """按图片和排序后的检测逐项比较类别、分数与框坐标."""
    cpu_images = cpu["images"]
    neuron_images = neuron["images"]
    if len(cpu_images) != len(neuron_images):
        raise ValueError("CPU 与 Neuron 图片数量不同.")

    max_score_delta = 0.0
    max_bbox_delta = 0.0
    image_summaries = []
    for cpu_image, neuron_image in zip(cpu_images, neuron_images, strict=True):
        if cpu_image["image"] != neuron_image["image"]:
            raise ValueError("CPU 与 Neuron 图片顺序不同.")
        cpu_detections = cpu_image["detections"]
        neuron_detections = neuron_image["detections"]
        if len(cpu_detections) != len(neuron_detections):
            raise ValueError(f"检测数量不同: {cpu_image['image']}.")
        for cpu_detection, neuron_detection in zip(
            cpu_detections, neuron_detections, strict=True
        ):
            if cpu_detection["class_id"] != neuron_detection["class_id"]:
                raise ValueError(f"检测类别不同: {cpu_image['image']}.")
            max_score_delta = max(
                max_score_delta,
                abs(cpu_detection["score"] - neuron_detection["score"]),
            )
            for cpu_value, neuron_value in zip(
                cpu_detection["bbox_xyxy"],
                neuron_detection["bbox_xyxy"],
                strict=True,
            ):
                max_bbox_delta = max(
                    max_bbox_delta,
                    abs(cpu_value - neuron_value),
                )
        image_summaries.append(
            {
                "image": cpu_image["image"],
                "detection_count": len(cpu_detections),
            }
        )

    return {
        "images": image_summaries,
        "max_score_delta": max_score_delta,
        "max_bbox_delta_px": max_bbox_delta,
    }


def main() -> None:
    """执行比较,写入报告并实施阈值门禁."""
    args = parse_args()
    result = compare_reports(load_report(args.cpu), load_report(args.neuron))
    result["max_score_delta_limit"] = args.max_score_delta
    result["max_bbox_delta_px_limit"] = args.max_bbox_delta
    result["passed"] = (
        result["max_score_delta"] <= args.max_score_delta
        and result["max_bbox_delta_px"] <= args.max_bbox_delta
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit("CPU/Neuron 检测差异超过阈值.")


if __name__ == "__main__":
    main()
