"""从真实板端 COCO 预测生成三个固定公开示例的检测效果."""

import argparse
import json
from pathlib import Path

import cv2


def main():
    """只绘制指定样例,记录图片地址和许可证的清单随输入保留."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.input_dir / "samples.json").read_text(encoding="utf-8"))
    selected = {sample["image_id"]: [] for sample in manifest["samples"]}
    with (args.work_dir / "predictions.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["image_id"] in selected and row["score"] >= 0.25:
                selected[row["image_id"]].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(manifest["samples"], 1):
        image = cv2.imread(str(args.input_dir / sample["file"]))
        if image is None:
            raise ValueError(f"示例图片无法读取: {sample['file']}.")
        for row in selected[sample["image_id"]]:
            x, y, width, height = row["bbox"]
            x1, y1, x2, y2 = map(round, (x, y, x + width, y + height))
            label = manifest["categories"][str(row["category_id"])]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 0), 2)
            cv2.putText(image, f"{label} {row['score']:.2f}",
                        (max(0, x1), max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 180, 0), 2, cv2.LINE_AA)
        output = args.output_dir / f"sample_{index}_detections.jpg"
        if not cv2.imwrite(str(output), image):
            raise RuntimeError(f"无法保存示例: {output}.")
        print(f"[示例 {index}/3] {output}", flush=True)


if __name__ == "__main__":
    main()
