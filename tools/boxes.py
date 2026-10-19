"""不依赖模型和芯片的 xyxy 矩形框计算."""

import numpy as np


def box_iou(box, boxes):
    """计算 xyxy 框的交并比."""
    intersection = np.maximum(
        np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2]), 0
    ).prod(axis=1)
    union = (np.maximum(box[2:] - box[:2], 0).prod()
             + np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
             - intersection)
    return intersection / np.maximum(union, 1e-9)
