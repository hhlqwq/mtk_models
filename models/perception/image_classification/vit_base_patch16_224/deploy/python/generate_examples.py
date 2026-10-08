"""从三张公开图片的 ViT 板端输出生成可视化示例."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_labels(path: Path) -> list[str]:
    """读取 ImageNet 类别名称."""
    return path.read_text(encoding="utf-8").splitlines()


def render_example(image: np.ndarray, topk: list[dict]) -> np.ndarray:
    """在原图下方添加 Top-5 分类结果面板."""
    panel_height = 155
    result = cv2.copyMakeBorder(
        image, 0, panel_height, 0, 0, cv2.BORDER_CONSTANT,
        value=(255, 255, 255))
    start_y = image.shape[0] + 28
    for rank, item in enumerate(topk, start=1):
        text = (f"Top {rank}: {item['class_name']} "
                f"(id={item['class_id']}, logit={item['logit']:.4f})")
        cv2.putText(result, text, (12, start_y + (rank - 1) * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1,
                    cv2.LINE_AA)
    return result


def load_raw_output(path: Path, shape: list[int]) -> np.ndarray:
    """读取可能包含 16 元素行对齐的 MDLA INT8 输出."""
    raw = np.fromfile(path, dtype=np.int8)
    classes = shape[-1]
    plain_size = int(np.prod(shape))
    padded_size = shape[0] * ((classes + 15) // 16 * 16)
    if raw.size == plain_size:
        return raw.reshape(shape)
    if raw.size == padded_size and len(shape) <= 2:
        return raw.reshape(shape[0], -1)[:, :classes].copy()
    raise ValueError(f"NPU 输出大小错误: {path}, 实际 {raw.size}")


def generate(args: argparse.Namespace) -> None:
    """读取三份真实 NPU 输出并生成图片、单图 JSON 和汇总清单."""
    labels = load_labels(args.labels)
    images = sorted(
        path for path in args.input_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:args.count]
    if len(images) != args.count:
        raise ValueError(f"公开图片数量错误: 需要 {args.count}, 实际 {len(images)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for position, source in enumerate(images, start=1):
        metadata_path = args.metadata_dir / f"sample_{position}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        detail = metadata["outputs"][0]
        output_path = args.npu_dir / f"sample_{position}_0.bin"
        raw = load_raw_output(output_path, detail["shape"]).reshape(-1)
        logits = ((raw.astype(np.float32) - detail["zero_point"])
                  * detail["scale"])
        indices = np.argsort(logits)[::-1][:5]
        topk = [{
            "class_id": int(index),
            "class_name": labels[index],
            "logit": float(logits[index]),
        } for index in indices]
        result = {
            "sample": position,
            "source_image": source.name,
            "board_output": output_path.name,
            "top5": topk,
        }
        json_path = args.output_dir / f"sample_{position}_top5.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"无法读取图片: {source}")
        rendered = render_example(image, topk)
        visual_path = args.output_dir / f"sample_{position}_top5.jpg"
        if not cv2.imwrite(str(visual_path), rendered):
            raise ValueError(f"无法写入图片: {visual_path}")
        summary.append(result)
        print(f"[EXAMPLE] {position}/{len(images)} {source.name}")
    (args.output_dir / "results.json").write_text(
        json.dumps({"samples": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[OK] ViT 三图示例: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
