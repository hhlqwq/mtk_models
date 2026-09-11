"""从正式 ViT 板端评测结果生成五张可视化示例."""

import argparse
import json
import shutil
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


def generate(args: argparse.Namespace) -> None:
    """读取五份真实 NPU 输出并生成图片、单图 JSON 和汇总清单."""
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    detail = metadata["outputs"][0]
    labels = load_labels(args.labels)
    records = [json.loads(line) for line in
               args.manifest.read_text(encoding="utf-8").splitlines()
               if line.strip()][:args.count]
    args.input_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for position, record in enumerate(records, start=1):
        source = args.images_dir / record["image"]
        copied = args.input_dir / f"sample_{position}{source.suffix.lower()}"
        shutil.copy2(source, copied)
        output_path = args.npu_dir / f"{record['stem']}_0.bin"
        raw = np.fromfile(output_path, dtype=np.int8)[:1000]
        if raw.size != 1000:
            raise ValueError(f"NPU 输出大小错误: {output_path}")
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
            "source_image": record["image"],
            "board_output": output_path.name,
            "top5": topk,
        }
        json_path = args.output_dir / f"sample_{position}_top5.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        image = cv2.imread(str(copied))
        if image is None:
            raise ValueError(f"无法读取图片: {copied}")
        rendered = render_example(image, topk)
        visual_path = args.output_dir / f"sample_{position}_top5.jpg"
        if not cv2.imwrite(str(visual_path), rendered):
            raise ValueError(f"无法写入图片: {visual_path}")
        summary.append(result)
        print(f"[EXAMPLE] {position}/{len(records)} {record['image']}")
    (args.output_dir / "results.json").write_text(
        json.dumps({"samples": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[OK] ViT 五图示例: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--npu-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
