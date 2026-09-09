"""比较 C++ 板端裁剪输入与 Python 参考预处理结果."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtmpose_utils import preprocess_image

INPUT_SHAPE = (1, 3, 256, 192)
INPUT_SCALE = 1.0
INPUT_ZERO_POINT = -128


def compare(args: argparse.Namespace) -> None:
    """重新生成量化输入并报告逐元素差异."""
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"无法读取图片: {args.image}")
    tensor, _ = preprocess_image(image, tuple(args.bbox))
    reference = np.clip(
        np.round(tensor / INPUT_SCALE) + INPUT_ZERO_POINT,
        -128, 127).astype(np.int8)
    actual = np.fromfile(args.actual, dtype=np.int8)
    if actual.size != int(np.prod(INPUT_SHAPE)):
        raise ValueError(f"C++ 输入大小异常: {actual.size}")
    actual = actual.reshape(INPUT_SHAPE)
    difference = np.abs(
        reference.astype(np.int16) - actual.astype(np.int16))
    mismatch = int(np.count_nonzero(difference))
    print(f"elements={difference.size}")
    print(f"mismatch={mismatch}")
    print(f"match_ratio={1.0 - mismatch / difference.size:.9f}")
    print(f"mean_abs={difference.mean():.9f}")
    print(f"max_abs={difference.max()}")
    if mismatch:
        raise RuntimeError("C++ 和 Python 预处理输入不一致.")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--bbox", type=float, nargs=4, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    compare(parse_args())
