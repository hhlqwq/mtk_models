#!/usr/bin/env python3
"""YOLO-World XL 图像预处理、检测头解码和 NMS 工具。"""

from dataclasses import dataclass

import cv2
import numpy as np


COCO_CLASSES = (
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
    "scissors", "teddy bear", "hair drier", "toothbrush",
)
STRIDES = (8, 16, 32)


@dataclass(frozen=True)
class ImageTransform:
    """保存方形填充和缩放参数。"""

    scale: float
    pad_x: int
    pad_y: int
    original_width: int
    original_height: int


def preprocess_image(image_bgr: np.ndarray) -> tuple[np.ndarray, ImageTransform]:
    """按上游导出 Demo 的黑边方形填充方式生成 NCHW RGB 输入。"""
    height, width = image_bgr.shape[:2]
    square_size = max(height, width)
    pad_y = (square_size - height) // 2
    pad_x = (square_size - width) // 2
    square = np.zeros((square_size, square_size, 3), dtype=np.uint8)
    square[pad_y:pad_y + height, pad_x:pad_x + width] = image_bgr
    resized = cv2.resize(square, (640, 640), interpolation=cv2.INTER_LINEAR)
    tensor = resized[:, :, ::-1].astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
    transform = ImageTransform(
        scale=640.0 / square_size,
        pad_x=pad_x,
        pad_y=pad_y,
        original_width=width,
        original_height=height,
    )
    return tensor, transform


def sigmoid(values: np.ndarray) -> np.ndarray:
    """稳定计算 Sigmoid。"""
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def decode_level(
    class_logits: np.ndarray,
    distances: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把单尺度 YOLOv8 风格距离预测解码为输入图坐标框。"""
    _, class_count, height, width = class_logits.shape
    if class_count != len(COCO_CLASSES) or distances.shape != (1, 4, height, width):
        raise ValueError(
            f"检测头形状异常: cls={class_logits.shape}, box={distances.shape}。"
        )
    scores = sigmoid(class_logits[0].transpose(1, 2, 0).reshape(-1, class_count))
    box_distances = distances[0].transpose(1, 2, 0).reshape(-1, 4)
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32) + 0.5,
        np.arange(height, dtype=np.float32) + 0.5,
    )
    points = np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=1)
    boxes = np.empty_like(box_distances, dtype=np.float32)
    boxes[:, 0:2] = (points - box_distances[:, 0:2]) * stride
    boxes[:, 2:4] = (points + box_distances[:, 2:4]) * stride
    return scores, boxes


def box_iou(one_box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """计算一个框与一组框的 IoU。"""
    top_left = np.maximum(one_box[:2], boxes[:, :2])
    bottom_right = np.minimum(one_box[2:], boxes[:, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0.0, None), axis=1)
    one_area = np.prod(np.clip(one_box[2:] - one_box[:2], 0.0, None))
    box_areas = np.prod(np.clip(boxes[:, 2:] - boxes[:, :2], 0.0, None), axis=1)
    return intersection / np.maximum(one_area + box_areas - intersection, 1e-12)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """执行逐类别 NMS 并返回保留索引。"""
    kept: list[int] = []
    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        order = class_indices[np.argsort(scores[class_indices])[::-1]]
        while order.size:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break
            overlaps = box_iou(boxes[current], boxes[order[1:]])
            order = order[1:][overlaps <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def decode_outputs(
    outputs: list[np.ndarray],
    transform: ImageTransform,
    score_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> list[dict[str, object]]:
    """解码三尺度输出、执行 NMS 并映射回原图。"""
    if len(outputs) != 6:
        raise ValueError(f"期望 6 个输出，实际为 {len(outputs)}。")

    all_scores = []
    all_boxes = []
    for level, stride in enumerate(STRIDES):
        scores, boxes = decode_level(outputs[level * 2], outputs[level * 2 + 1], stride)
        all_scores.append(scores)
        all_boxes.append(boxes)
    scores = np.concatenate(all_scores, axis=0)
    boxes = np.concatenate(all_boxes, axis=0)

    point_indices, labels = np.nonzero(scores > score_threshold)
    if point_indices.size == 0:
        return []
    candidate_scores = scores[point_indices, labels]
    candidate_boxes = boxes[point_indices]
    keep = class_aware_nms(
        candidate_boxes, candidate_scores, labels, iou_threshold=iou_threshold
    )
    keep = keep[np.argsort(candidate_scores[keep])[::-1]][:max_detections]

    candidate_boxes = candidate_boxes[keep]
    candidate_scores = candidate_scores[keep]
    labels = labels[keep]
    candidate_boxes[:, 0::2] /= transform.scale
    candidate_boxes[:, 1::2] /= transform.scale
    candidate_boxes[:, 0::2] -= transform.pad_x
    candidate_boxes[:, 1::2] -= transform.pad_y
    candidate_boxes[:, 0::2] = np.clip(
        candidate_boxes[:, 0::2], 0, transform.original_width
    )
    candidate_boxes[:, 1::2] = np.clip(
        candidate_boxes[:, 1::2], 0, transform.original_height
    )

    detections = []
    for box, score, label in zip(
        candidate_boxes, candidate_scores, labels, strict=True
    ):
        detections.append(
            {
                "class_id": int(label),
                "class_name": COCO_CLASSES[int(label)],
                "score": float(score),
                "bbox_xyxy": [float(value) for value in box],
            }
        )
    return detections


def render_detections(
    image_bgr: np.ndarray,
    detections: list[dict[str, object]],
) -> np.ndarray:
    """在图像上绘制检测框、类别和置信度。"""
    rendered = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
        label = f"{detection['class_name']} {detection['score']:.2f}"
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (40, 220, 40), 2)
        cv2.putText(
            rendered,
            label,
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 220, 40),
            1,
            cv2.LINE_AA,
        )
    return rendered
