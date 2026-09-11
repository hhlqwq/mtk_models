"""将 YOLOv5s 板端 C++ 预测绘制到三张公开样例图片."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

COCO_NAMES = (
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
    "scissors", "teddy bear", "hair drier", "toothbrush")
COCO_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78,
    79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90)
COCO_CATEGORY_NAMES = dict(zip(COCO_CATEGORY_IDS, COCO_NAMES))


def load_predictions(path: Path) -> dict[int, list[dict]]:
    """按 image_id 汇总板端 JSONL 检测结果."""
    grouped = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        grouped[int(item["image_id"])].append(item)
    return grouped


def render(args: argparse.Namespace) -> None:
    """生成检测可视化、单图 JSON 和汇总结果."""
    predictions = load_predictions(args.predictions)
    images = sorted(
        path for path in args.input_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:args.count]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for position, image_path in enumerate(images, start=1):
        image_id = int(image_path.stem)
        detections = predictions.get(image_id, [])
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        for detection in detections:
            x, y, width, height = detection["bbox"]
            category_id = int(detection["category_id"])
            score = float(detection["score"])
            cv2.rectangle(image, (round(x), round(y)),
                          (round(x + width), round(y + height)),
                          (0, 255, 0), 2, cv2.LINE_AA)
            label = f"{COCO_CATEGORY_NAMES[category_id]} {score:.2f}"
            cv2.putText(image, label, (round(x), max(20, round(y) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                        cv2.LINE_AA)
        record = {"sample": position, "image_id": image_id,
                  "image": image_path.name, "detections": detections}
        (args.output_dir / f"sample_{position}_detections.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        visual = args.output_dir / f"sample_{position}_detections.jpg"
        if not cv2.imwrite(str(visual), image):
            raise ValueError(f"无法写入图片: {visual}")
        summary.append(record)
        print(f"[EXAMPLE] {position}/{len(images)} {image_path.name}, "
              f"检测 {len(detections)} 个目标")
    (args.output_dir / "results.json").write_text(
        json.dumps({"samples": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[OK] YOLOv5s 三图示例: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
