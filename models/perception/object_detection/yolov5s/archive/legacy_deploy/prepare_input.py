"""为 YOLOv5s 板端推理准备 NCHW INT8 输入."""

import argparse
import json
from pathlib import Path

import cv2
import mtk_converter
import numpy as np


def letterbox(image: np.ndarray, image_size: int) -> tuple[np.ndarray, float,
                                                            int, int]:
    """按 YOLOv5 规则缩放并填充图片."""
    height, width = image.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height))
    canvas = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas, scale, left, top


def prepare_input(args: argparse.Namespace) -> None:
    """读取图片并按 TFLite 量化参数写入板端输入."""
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"无法读取图片: {args.image}")
    parser = mtk_converter.TFLiteParser(str(args.tflite))
    input_detail = parser.get_input_tensor_details()[0]
    output_details = parser.get_output_tensor_details()
    input_scale = input_detail["quantization"]["scales"][0]
    input_zero_point = input_detail["quantization"]["zero_points"][0]

    canvas, resize_scale, left, top = letterbox(image, args.image_size)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    nchw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    quantized = np.clip(
        np.round(nchw / input_scale) + input_zero_point, -128,
        127).astype(np.int8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantized[np.newaxis].tofile(args.output)
    metadata = {
        "source_image": str(args.image),
        "original_shape": list(image.shape[:2]),
        "input_shape": input_detail["shape"],
        "resize_scale": resize_scale,
        "pad_left": left,
        "pad_top": top,
        "input_quantization": {
            "scale": input_scale,
            "zero_point": input_zero_point,
        },
        "outputs": [{
            "name": detail["name"],
            "shape": detail["shape"],
            "scale": detail["quantization"]["scales"][0],
            "zero_point": detail["quantization"]["zero_points"][0],
        } for detail in output_details],
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 板端输入: {args.output}")
    print(f"[OK] 张量元数据: {args.metadata}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--tflite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_input(parse_args())
