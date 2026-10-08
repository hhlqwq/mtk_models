"""将 MMPose 人体检测框转换为板端 C++ 评测清单."""

import argparse
import json
from pathlib import Path

def prepare_manifest(args: argparse.Namespace) -> None:
    """校验人体检测结果并写出按图片排序的 TSV 清单."""
    detections = json.loads(args.detections.read_text(encoding="utf-8"))
    records = []
    for detection_id, item in enumerate(detections):
        bbox = item.get("bbox", [])
        if (int(item.get("category_id", -1)) != 1 or len(bbox) != 4 or
                float(bbox[2]) <= 0 or float(bbox[3]) <= 0):
            raise ValueError(f"无效人体检测框,index={detection_id}: {item}")
        records.append((
            int(item["image_id"]),
            detection_id,
            float(item["score"]),
            *(float(value) for value in bbox),
        ))
        if ((detection_id + 1) % 10000 == 0 or
                detection_id + 1 == len(detections)):
            print(f"[MANIFEST] {detection_id + 1}/{len(detections)}")
    records.sort(key=lambda value: (value[0], value[1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        output.write("image_id\tdetection_id\tbbox_score\tx\ty\tw\th\n")
        for record in records:
            output.write("\t".join(map(str, record)) + "\n")
    print(f"[OK] 人体检测框清单: {args.output},共 {len(records)} 个框.")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_manifest(parse_args())
