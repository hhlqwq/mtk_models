"""将 ViT-Base Patch16 224 FP32 ONNX 量化为 MTK INT8 TFLite。"""

import argparse
from collections.abc import Iterator
from pathlib import Path

import cv2
import mtk_converter
import numpy as np
import tqdm

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_path: Path, crop_size: int = 224,
                     resize_size: int = 256) -> np.ndarray:
    """按 ImageNet 标准评估预处理生成 NCHW FP32 输入。"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
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


def calibration_data(
        calibration_dir: Path, sample_count: int) -> Iterator[list[np.ndarray]]:
    """按固定顺序生成校准数据, 带进度提示。"""
    image_paths = sorted(
        path for path in calibration_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPEG"})
    selected_paths = image_paths[:sample_count]
    if len(selected_paths) < sample_count:
        raise ValueError(
            f"校准图片不足: 需要 {sample_count}, 实际 {len(selected_paths)}")
    for image_path in tqdm.tqdm(selected_paths, desc="准备 ViT 校准数据"):
        yield [preprocess_image(image_path)]


def convert_model(args: argparse.Namespace) -> None:
    """创建 MTK ONNX Converter 并执行 INT8 PTQ。"""
    converter = mtk_converter.OnnxConverter.from_model_proto_file(
        str(args.onnx))
    converter.quantize = True
    converter.calibration_data_gen = lambda: calibration_data(
        args.calibration_dir, args.samples)
    # ViT attention 对激活离群值敏感, 默认开启 per-channel 权重量化。
    converter.use_per_output_channel_quantization = True
    converter.convert_to_tflite(str(args.output))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    convert_model(parse_args())
