"""解码 YOLOv5s 板端 INT8 输出并绘制检测框。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ANCHORS = np.array([
    [[1.25, 1.625], [2.0, 3.75], [4.125, 2.875]],
    [[1.875, 3.8125], [3.875, 2.8125], [3.6875, 7.4375]],
    [[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]],
], dtype=np.float32)
STRIDES = np.array([8.0, 16.0, 32.0], dtype=np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """计算稳定的 Sigmoid。"""
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def decode_head(values: np.ndarray, head_index: int) -> np.ndarray:
    """将单个检测头解码为 xywh、置信度和类别概率。"""
    _, _, height, width = values.shape
    values = values.reshape(1, 3, 85, height, width)
    values = values.transpose(0, 1, 3, 4, 2)
    grid_y, grid_x = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32), indexing="ij")
    grid = np.stack((grid_x, grid_y), axis=-1)[None, None] - 0.5
    activated = sigmoid(values)
    xy = (activated[..., :2] * 2.0 + grid) * STRIDES[head_index]
    anchor_grid = (ANCHORS[head_index] * STRIDES[head_index])[
        None, :, None, None, :]
    wh = (activated[..., 2:4] * 2.0)**2 * anchor_grid
    decoded = np.concatenate((xy, wh, activated[..., 4:]), axis=-1)
    return decoded.reshape(-1, 85)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """计算一个框与多个框的 IoU。"""
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0.0), axis=1)
    box_area = np.prod(np.maximum(box[2:] - box[:2], 0.0))
    boxes_area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0.0),
                         axis=1)
    return intersection / np.maximum(box_area + boxes_area - intersection,
                                     1e-9)


def class_aware_nms(boxes: np.ndarray, scores: np.ndarray,
                    classes: np.ndarray, threshold: float) -> list[int]:
    """执行逐类别 NMS。"""
    kept: list[int] = []
    for class_id in np.unique(classes):
        indices = np.where(classes == class_id)[0]
        indices = indices[np.argsort(scores[indices])[::-1]]
        while indices.size:
            current = int(indices[0])
            kept.append(current)
            if indices.size == 1:
                break
            ious = box_iou(boxes[current], boxes[indices[1:]])
            indices = indices[1:][ious <= threshold]
    return sorted(kept, key=lambda index: scores[index], reverse=True)


def postprocess(args: argparse.Namespace) -> None:
    """读取板端输出、解码、执行 NMS 并写入结果。"""
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    decoded_heads = []
    for index, detail in enumerate(metadata["outputs"]):
        output_path = args.output_dir / f"output_{index}.bin"
        quantized = np.fromfile(output_path, dtype=np.int8)
        expected_size = int(np.prod(detail["shape"]))
        if quantized.size != expected_size:
            raise ValueError(
                f"输出大小错误: {output_path}, 期望 {expected_size}, "
                f"实际 {quantized.size}")
        values = quantized.reshape(detail["shape"]).astype(np.float32)
        values = (values - detail["zero_point"]) * detail["scale"]
        decoded_heads.append(decode_head(values, index))

    predictions = np.concatenate(decoded_heads, axis=0)
    class_probabilities = predictions[:, 5:]
    classes = np.argmax(class_probabilities, axis=1)
    scores = predictions[:, 4] * class_probabilities[
        np.arange(predictions.shape[0]), classes]
    selected = scores >= args.confidence
    predictions = predictions[selected]
    classes = classes[selected]
    scores = scores[selected]

    boxes = np.empty((predictions.shape[0], 4), dtype=np.float32)
    boxes[:, :2] = predictions[:, :2] - predictions[:, 2:4] / 2.0
    boxes[:, 2:] = predictions[:, :2] + predictions[:, 2:4] / 2.0
    kept = class_aware_nms(boxes, scores, classes, args.iou)
    boxes, scores, classes = boxes[kept], scores[kept], classes[kept]

    original_height, original_width = metadata["original_shape"]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - metadata["pad_left"])
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - metadata["pad_top"])
    boxes /= metadata["resize_scale"]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_height)

    detections = [{
        "class_id": int(class_id),
        "score": float(score),
        "box_xyxy": [float(value) for value in box],
    } for box, score, class_id in zip(boxes, scores, classes)]
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(detections, ensure_ascii=False, indent=2), encoding="utf-8")

    image = cv2.imread(metadata["source_image"])
    if image is None:
        raise ValueError(f"无法读取原始图片: {metadata['source_image']}")
    for detection in detections:
        x1, y1, x2, y2 = map(round, detection["box_xyxy"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{detection['class_id']} {detection['score']:.3f}"
        cv2.putText(image, label, (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(args.image_output), image)
    print(f"[OK] 检测数量: {len(detections)}")
    print(f"[OK] JSON: {args.result}")
    print(f"[OK] 图片: {args.image_output}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--image-output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


if __name__ == "__main__":
    postprocess(parse_args())
