"""MobileFaceNet 的对齐人脸输入处理。"""

from pathlib import Path

import cv2
import numpy as np


def load_aligned_face(path: Path) -> np.ndarray:
    """按上游训练协议生成 BGR NCHW 浮点输入。"""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
    normalized = (image.astype(np.float32) - 127.5) / 128.0
    return np.transpose(normalized, (2, 0, 1))[None, ...].copy()
