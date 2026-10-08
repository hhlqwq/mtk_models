"""从本次全量板端预测生成少量效果示例,不重新执行模型推理."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


BODY_LINKS = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7),
              (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
              (11, 13), (13, 15), (12, 14), (14, 16))


def read_json(path: Path):
    """读取 UTF-8 JSON 文件."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    """逐行读取预测,避免为少量示例载入整个全量结果."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_image(path: Path) -> np.ndarray:
    """读取已选定的示例输入."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"无法读取示例图片: {path}.")
    return image


def panel(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """缩放展示图片并在底部增加英文结果说明."""
    width = 720
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    image = cv2.resize(image, (width, height))
    result = cv2.copyMakeBorder(image, 0, 35 * len(lines) + 15, 0, 0,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for index, line in enumerate(lines):
        cv2.putText(result, line, (12, height + 30 + 35 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1,
                    cv2.LINE_AA)
    return result


def save_image(args, position: int, suffix: str, image: np.ndarray) -> None:
    """保存单张可视化,每个样本只保留一张效果图."""
    destination = args.output_dir / f"sample_{position}_{suffix}.jpg"
    if not cv2.imwrite(str(destination), image):
        raise ValueError(f"无法写入示例: {destination}.")
    print(f"[EXAMPLE] {position}: {destination.name}", flush=True)


def classification(args, manifest) -> None:
    """复用 ViT 类别排序展示 Top-1 与 Top-5,不虚构置信度."""
    samples = manifest["samples"]
    wanted = {Path(item["image"]).name for item in samples}
    records = {}
    for row in read_jsonl(args.work_dir / "predictions.jsonl"):
        if row["image"] in wanted:
            records[row["image"]] = row
        if len(records) == len(wanted):
            break
    labels = manifest["class_names"]
    for position, item in enumerate(samples, 1):
        record = records[Path(item["image"]).name]
        lines = ["Board prediction: Top-5", f"Ground truth: {item['label']}"]
        lines += [f"{rank}. {labels[class_id]}" for rank, class_id in
                  enumerate(record["top5"], 1)]
        save_image(args, position, "classification",
                   panel(load_image(args.input_dir / item["file"]), lines))


def detections(args, manifest) -> None:
    """复用 COCO 检测结果,按展示阈值绘制类别和检测框."""
    samples = manifest["samples"]
    selected = {item["image_id"]: [] for item in samples}
    if args.model == "yolov5s":
        for row in read_jsonl(args.work_dir / "predictions.jsonl"):
            if row["image_id"] in selected:
                x, y, width, height = row["bbox"]
                selected[row["image_id"]].append({
                    "bbox": [x, y, x + width, y + height],
                    "score": row["score"],
                    "label": manifest["categories"][str(row["category_id"])],
                })
    else:
        wanted = {Path(item["image"]).name: item["image_id"] for item in samples}
        for row in read_jsonl(args.work_dir / "raw/results.jsonl"):
            if row["image"] in wanted:
                selected[wanted[row["image"]]] = [
                    {"bbox": box["bbox_xyxy"], "score": box["score"],
                     "label": manifest["class_names"][box["class_id"]]}
                    for box in row["detections"]]
    for position, item in enumerate(samples, 1):
        image = load_image(args.input_dir / item["file"])
        count = 0
        for box in selected[item["image_id"]]:
            if box["score"] < 0.25:
                continue
            x1, y1, x2, y2 = map(round, box["bbox"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 0), 2)
            cv2.putText(image, f"{box['label']} {box['score']:.2f}",
                        (max(0, x1), max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 180, 0), 2, cv2.LINE_AA)
            count += 1
        save_image(args, position, "detections", panel(
            image, [f"Board prediction: {count} detections (score >= 0.25)"]))


def pose(args, manifest) -> None:
    """展示完整人体、手部骨架和 WholeBody 关节点."""
    selected = {item["image_id"]: [] for item in manifest["samples"]}
    for row in read_jsonl(args.work_dir / "predictions.jsonl"):
        if row["image_id"] in selected and row["bbox_score"] >= 0.3:
            selected[row["image_id"]].append(row)
    links = list(BODY_LINKS)
    for base in (91, 112):
        for finger in range(5):
            start = base + 1 + finger * 4
            links += [(base, start), (start, start + 1),
                      (start + 1, start + 2), (start + 2, start + 3)]
    for position, item in enumerate(manifest["samples"], 1):
        image = load_image(args.input_dir / item["file"])
        people = sorted(selected[item["image_id"]],
                        key=lambda row: row["bbox_score"], reverse=True)[:10]
        for row in people:
            points = np.asarray(row["keypoints"]).reshape(-1, 3)
            for left, right in links:
                if min(points[left, 2], points[right, 2]) >= 0.2:
                    cv2.line(image, tuple(np.rint(points[left, :2]).astype(int)),
                             tuple(np.rint(points[right, :2]).astype(int)),
                             (0, 180, 0), 2, cv2.LINE_AA)
            for x, y, score in points:
                if score >= 0.2:
                    cv2.circle(image, (round(x), round(y)), 2, (0, 100, 255), -1)
        save_image(args, position, "keypoints", panel(
            image, ["Board prediction: WholeBody keypoints and skeleton"]))


def segmentation(args, manifest) -> None:
    """从选定样本的 RLE 检查点恢复实例分割可视化."""
    from pycocotools import mask as mask_utils

    colors = ((40, 180, 255), (255, 120, 40), (80, 220, 80),
              (200, 80, 200), (255, 220, 60), (80, 100, 255))
    for position, item in enumerate(manifest["samples"], 1):
        record = read_json(args.work_dir / "predictions_by_image" /
                           f"{Path(item['image']).stem}.json")
        image = load_image(args.input_dir / item["file"])
        count = 0
        for prediction in sorted(record["predictions"], key=lambda x: x["score"]):
            if prediction["score"] < 0.25:
                continue
            mask = mask_utils.decode(prediction["segmentation"]).astype(bool)
            color = np.array(colors[count % len(colors)], dtype=np.float32)
            image[mask] = (image[mask] * 0.55 + color * 0.45).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, tuple(map(int, color)), 1)
            count += 1
        save_image(args, position, "segmentation", panel(
            image, [f"Board prediction: {count} masks (score >= 0.25)"]))


def depth(args, manifest) -> None:
    """复用深度输出展示原图与相对深度,颜色较暖表示更近."""
    from full_accuracy_board import load_output

    metadata = read_json(args.models_dir / "model_int8.json")
    paths = sorted(read_json(args.dataset_root / "annotations.json"))
    indices = {relative: index for index, relative in enumerate(paths, 1)}
    for position, item in enumerate(manifest["samples"], 1):
        values = load_output(args.work_dir / "outputs" /
                             f"{indices[item['image']]:05d}.bin", metadata)
        normalized = (values - values.min()) / (values.max() - values.min())
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8),
                                    cv2.COLORMAP_TURBO)
        image = load_image(args.input_dir / item["file"])
        colored = cv2.resize(colored, (image.shape[1], image.shape[0]))
        save_image(args, position, "depth", panel(
            np.concatenate([image, colored], axis=1),
            ["Input | Board relative depth (warm: near, cool: far)"]))


def face(args, manifest) -> None:
    """展示真实十折验证的余弦相似度与同人判定."""
    wanted = {str(item["pair_id"]) for item in manifest["samples"]}
    records = {str(row["pair_id"]): row for row in
               read_jsonl(args.work_dir / "report/pair_results.jsonl")
               if str(row["pair_id"]) in wanted}
    for position, item in enumerate(manifest["samples"], 1):
        record = records[str(item["pair_id"])]
        images = [cv2.resize(load_image(args.input_dir / item[field]), (280, 280))
                  for field in ("file_a", "file_b")]
        predicted = "same person" if record["predicted_same"] else "different people"
        truth = "same person" if item["is_same"] else "different people"
        lines = [f"Board prediction: {predicted}", f"Ground truth: {truth}",
                 f"Cosine similarity: {record['cosine_similarity']:.4f}",
                 "Decision threshold: trained on the other 9 LFW folds"]
        save_image(args, position, "verification",
                   panel(np.concatenate(images, axis=1), lines))


def speech(args, manifest) -> None:
    """解码两段真实板端 Token,只保留可读的识别文本."""
    from whisper.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(multilingual=True, language="en", task="transcribe")
    wanted = {item["sample_id"] for item in manifest["samples"]}
    records = {row["sample_id"]: row for row in
               read_jsonl(args.work_dir / "board_predictions.jsonl")
               if row["sample_id"] in wanted}
    lines = ["# 板端语音识别示例", ""]
    for position, item in enumerate(manifest["samples"], 1):
        row = records[item["sample_id"]]
        if row.get("status") != "ok":
            raise ValueError(f"语音示例推理未成功: {item['sample_id']}.")
        text = tokenizer.decode(row["tokens"]).strip()
        lines += [f"## 示例 {position}", "", f"输入: `{item['file']}`", "",
                  f"参考文本: {item['reference']}", "", f"板端识别: {text}", ""]
        print(f"[EXAMPLE] {position}: {item['sample_id']}", flush=True)
    (args.output_dir / "transcripts.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main() -> None:
    """统一选择任务适配器,仅保存固定的少量示例产物."""
    adapters = {"vit_base_patch16_224": classification, "yolov5s": detections,
                "yoloworld_xl": detections, "rtmpose_body2d": pose,
                "fastsam": segmentation, "depth_anything_v2_small": depth,
                "mobilefacenet": face, "whisper_tiny": speech}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=adapters, required=True)
    for name in ("input-dir", "output-dir", "work-dir", "dataset-root", "models-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.input_dir / "samples.json")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    adapters[args.model](args, manifest)
    print(f"[OK] 板端效果示例: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
