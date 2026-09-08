"""解码 ViT 板端原生 INT8 logits 输出, 给出 top-1/top-5。"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_raw_output(output_path: Path, shape: list[int]) -> np.ndarray:
    """读取 MDLA 原生输出并还原为声明 shape (行 stride 16 对齐容错)。"""
    quantized = np.fromfile(output_path, dtype=np.int8)
    n, classes = shape[0], shape[-1]
    plain = int(np.prod(shape))
    pad = (classes + 15) // 16 * 16
    padded = n * pad
    if quantized.size == plain:
        return quantized.reshape(shape)
    if quantized.size == padded and len(shape) <= 2:
        return quantized.reshape(n, pad)[:, :classes].copy()
    raise ValueError(
        f"输出大小错误: {output_path}, 期望 {plain} 或 {padded}, "
        f"实际 {quantized.size}")


def postprocess(args: argparse.Namespace) -> None:
    """反量化 logits 并打印/保存 top-k 结果。"""
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    detail = metadata["outputs"][0]
    raw = load_raw_output(args.output_dir / "output_0.bin", detail["shape"])
    logits = (raw.astype(np.float32) - detail["zero_point"]) * detail["scale"]
    logits = logits.reshape(-1)
    top_indices = np.argsort(logits)[::-1][:args.topk]
    names = None
    if args.labels is not None and args.labels.exists():
        names = args.labels.read_text(encoding="utf-8").splitlines()
    result = {
        "source_image": metadata["source_image"],
        "topk": [{
            "class_id": int(index),
            "class_name": (names[index] if names and
                           index < len(names) else ""),
            "logit": float(logits[index]),
        } for index in top_indices],
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in result["topk"]:
        print(f"top{item['class_id']:>5} {item['class_name']:<28} "
              f"logit={item['logit']:.4f}")
    print(f"[OK] 结果: {args.result}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--labels", type=Path, default=None,
                        help="可选类名表 (Qualcomm 归档内 labels.txt)。")
    return parser.parse_args()


if __name__ == "__main__":
    postprocess(parse_args())
