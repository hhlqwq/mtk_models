"""为 ViT 板端推理准备 NCHW INT8 输入."""

import argparse
import json
from pathlib import Path

import cv2
import mtk_converter
import numpy as np


def preprocess(image: np.ndarray, crop_size: int = 224,
               resize_size: int = 256) -> np.ndarray:
    """ImageNet 标准评估几何预处理, 返回 NCHW FP32 [0,1].

    Qualcomm 导出的 ONNX 已在图内完成 mean/std 归一化 (首节点 Sub/Div),
    外部输入必须是 rgb/255 的 [0,1] 范围, 不允许再次归一化.
    """
    height, width = image.shape[:2]
    scale = resize_size / min(height, width)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)),
                         interpolation=cv2.INTER_CUBIC)
    top = (resized.shape[0] - crop_size) // 2
    left = (resized.shape[1] - crop_size) // 2
    crop = resized[top:top + crop_size, left:left + crop_size]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    scaled = rgb.astype(np.float32) / 255.0
    return scaled.transpose(2, 0, 1)[np.newaxis].copy()


def prepare_input(args: argparse.Namespace) -> None:
    """读取图片并按 TFLite 量化参数写入板端输入与元数据."""
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"无法读取图片: {args.image}")
    parser = mtk_converter.TFLiteParser(str(args.tflite))
    input_detail = parser.get_input_tensor_details()[0]
    output_details = parser.get_output_tensor_details()
    if list(input_detail["shape"]) != [1, 3, 224, 224]:
        raise ValueError(f"TFLite 输入 shape 异常: {input_detail['shape']}")
    if len(output_details) != 1 or list(output_details[0]["shape"]) != [1, 1000]:
        raise ValueError(f"TFLite 输出结构异常: {output_details}")
    scale = float(input_detail["quantization"]["scales"][0])
    zero_point = int(input_detail["quantization"]["zero_points"][0])
    fp32 = preprocess(image)
    quantized = np.clip(np.round(fp32 / scale) + zero_point, -128,
                        127).astype(np.int8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantized.tofile(args.output)
    metadata = {
        "source_image": str(args.image),
        "input_shape": [int(value) for value in input_detail["shape"]],
        "input_quantization": {"scale": scale, "zero_point": zero_point},
        "outputs": [{
            "name": detail["name"],
            "shape": [int(value) for value in detail["shape"]],
            "scale": float(detail["quantization"]["scales"][0]),
            "zero_point": int(detail["quantization"]["zero_points"][0]),
        } for detail in output_details],
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 板端输入: {args.output}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--tflite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_input(parse_args())
