#!/usr/bin/env python3
"""比较官方与兼容 YOLO-World ONNX 的 CPU 输出。"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="验证 opset 转换前后数值等价性。")
    parser.add_argument("--source", type=Path, required=True, help="官方 ONNX 路径。")
    parser.add_argument("--converted", type=Path, required=True, help="兼容 ONNX 路径。")
    parser.add_argument("--report", type=Path, required=True, help="JSON 报告路径。")
    parser.add_argument("--seed", type=int, default=20260915, help="随机种子。")
    return parser.parse_args()


def create_session(path: Path) -> ort.InferenceSession:
    """创建禁用图优化的 CPU 会话。"""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def main() -> None:
    """运行相同随机输入并保存逐输出误差。"""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    image = rng.random((1, 3, 640, 640), dtype=np.float32)

    print("[1/3] 创建官方模型 CPU 会话。")
    source_session = create_session(args.source)
    print("[2/3] 创建兼容模型 CPU 会话。")
    converted_session = create_session(args.converted)
    source_outputs = source_session.run(None, {"images": image})
    converted_outputs = converted_session.run(None, {"images": image})

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

    report = {
        "schema_version": 1,
        "seed": args.seed,
        "input_shape": list(image.shape),
        "comparisons": comparisons,
        "all_outputs_exact": all(item["max_abs"] == 0.0 for item in comparisons),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("[3/3] 数值等价报告已写入。")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_outputs_exact"]:
        raise RuntimeError("opset 转换前后输出未达到逐元素完全一致。")


if __name__ == "__main__":
    main()
