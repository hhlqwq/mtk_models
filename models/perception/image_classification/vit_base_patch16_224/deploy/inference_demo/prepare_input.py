"""为 ViT 板端推理准备 NCHW INT8 输入。"""

import argparse
import json
from pathlib import Path

import cv2
import mtk_converter
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: np.ndarray, crop_size: int = 224,
               resize_size: int = 256) -> np.ndarray:
    """ImageNet 标准评估预处理, 返回 NCHW FP32。"""
    height, width = image.shape[:2]
    scale = resize_size / min(height, width)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)),
                         interpolation=cv2.INTER_CUBIC)
    top = (resized.shape[0] - crop_size) // 2
    left = (resized.shape[1] - crop_size) // 2
    crop = resized[top:top + crop_size, left:left + crop_size]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return normalized.transpose(2, 0, 1)[np.newaxis]


def prepare_input(args: argparse.Namespace) -> None:
    """读取图片并按 TFLite 量化参数写入板端输入与元数据。"""
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"无法读取图片: {args.image}")
    parser = mtk_converter.TFLiteParser(str(args.tflite))
    input_detail = parser.get_input_tensor_details()[0]
    output_details = parser.get_output_tensor_details()
    scale = input_detail["quantization"]["scales"][0]
    zero_point = input_detail["quantization"]["zero_points"][0]
    fp32 = preprocess(image)
    quantized = np.clip(np.round(fp32 / scale) + zero_point, -128,
                        127).astype(np.int8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantized.tofile(args.output)
    metadata = {
        "source_image": str(args.image),
        "input_shape": list(input_detail["shape"]),
        "input_quantization": {"scale": scale, "zero_point": zero_point},
        "outputs": [{
            "name": detail["name"],
            "shape": list(detail["shape"]),
            "scale": detail["quantization"]["scales"][0],
            "zero_point": detail["quantization"]["zero_points"][0],
        } for detail in output_details],
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 板端输入: {args.output}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--tflite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_input(parse_args())
