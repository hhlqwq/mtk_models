#!/usr/bin/env python3
"""比较官方与兼容 YOLO-World ONNX 的 CPU 输出."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description="验证 opset 转换前后数值等价性.")
    parser.add_argument("--source", type=Path, required=True, help="官方 ONNX 路径.")
    parser.add_argument("--converted", type=Path, required=True, help="兼容 ONNX 路径.")
    parser.add_argument("--raw", type=Path, required=True, help="Raw 检测头 ONNX 路径.")
    parser.add_argument("--report", type=Path, required=True, help="JSON 报告路径.")
    parser.add_argument("--seed", type=int, default=20260915, help="随机种子.")
    return parser.parse_args()


def create_session(path: Path) -> ort.InferenceSession:
    """创建禁用图优化的 CPU 会话."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def decode_dfl(raw_bbox: np.ndarray) -> np.ndarray:
    """把 64 通道 DFL logits 还原为官方模型的 4 通道距离输出."""
    batch, channels, height, width = raw_bbox.shape
    if batch != 1 or channels != 64:
        raise ValueError(f"DFL 输入形状异常: {raw_bbox.shape}.")
    logits = raw_bbox.reshape(batch, 4, 16, height * width)
    logits = logits.transpose(0, 3, 1, 2)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    bins = np.arange(16, dtype=np.float32)
    distances = (probabilities * bins).sum(axis=-1)
    return distances.transpose(0, 2, 1).reshape(batch, 4, height, width)


def main() -> None:
    """运行相同随机输入并保存逐输出误差."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    image = rng.random((1, 3, 640, 640), dtype=np.float32)

    print("[1/4] 创建官方模型 CPU 会话.")
    source_session = create_session(args.source)
    print("[2/4] 创建兼容模型与 Raw 检测头模型 CPU 会话.")
    converted_session = create_session(args.converted)
    raw_session = create_session(args.raw)
    source_outputs = source_session.run(None, {"images": image})
    converted_outputs = converted_session.run(None, {"images": image})
    raw_outputs = raw_session.run(None, {"images": image})

    comparisons = []
    for index, (source, converted) in enumerate(
        zip(source_outputs, converted_outputs, strict=True)
    ):
        difference = np.abs(source - converted)
        comparisons.append(
            {
                "index": index,
                "shape": list(source.shape),
                "max_abs": float(difference.max(initial=0.0)),
                "mean_abs": float(difference.mean()),
            }
        )

    raw_comparisons = []
    for level in range(3):
        source_class = source_outputs[level * 2]
        source_bbox = source_outputs[level * 2 + 1]
        raw_class = raw_outputs[level * 2]
        decoded_bbox = decode_dfl(raw_outputs[level * 2 + 1])
        class_difference = np.abs(source_class - raw_class)
        bbox_difference = np.abs(source_bbox - decoded_bbox)
        raw_comparisons.append(
            {
                "level": level,
                "class_max_abs": float(class_difference.max(initial=0.0)),
                "class_mean_abs": float(class_difference.mean()),
                "bbox_max_abs": float(bbox_difference.max(initial=0.0)),
                "bbox_mean_abs": float(bbox_difference.mean()),
            }
        )

    report = {
        "schema_version": 1,
        "seed": args.seed,
        "input_shape": list(image.shape),
        "comparisons": comparisons,
        "all_outputs_exact": all(item["max_abs"] == 0.0 for item in comparisons),
        "raw_head_comparisons": raw_comparisons,
        "raw_head_equivalent": all(
            item["class_max_abs"] == 0.0 and item["bbox_max_abs"] <= 1e-5
            for item in raw_comparisons
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("[4/4] 数值等价报告已写入.")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_outputs_exact"]:
        raise RuntimeError("opset 转换前后输出未达到逐元素完全一致.")
    if not report["raw_head_equivalent"]:
        raise RuntimeError("Raw 检测头 CPU 后处理与官方框距离输出不等价.")


if __name__ == "__main__":
    main()
