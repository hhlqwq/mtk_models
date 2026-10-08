"""将 RTMPose Body2d FP32 ONNX 量化为 MTK INT8 TFLite."""

import argparse
from collections.abc import Iterator
from pathlib import Path

import cv2
import mtk_converter
import numpy as np
from tqdm import tqdm

from rtmpose_utils import load_person_samples, preprocess_image


def calibration_data(
        image_dir: Path, annotations: Path, sample_count: int,
        offset: int) -> Iterator[list[np.ndarray]]:
    """从 COCO person 框生成固定且可复现的校准输入."""
    samples = load_person_samples(annotations)
    selected = samples[offset:offset + sample_count]
    if len(selected) < sample_count:
        raise ValueError(
            f"有效 person 框不足: 需要 {sample_count}, 实际 {len(selected)}")
    for sample in tqdm(selected, desc="准备 RTMPose 校准数据"):
        image_path = image_dir / sample["file_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取校准图片: {image_path}")
        input_tensor, _ = preprocess_image(image, sample["bbox"])
        yield [input_tensor]


def convert_model(args: argparse.Namespace) -> None:
    """创建 MTK ONNX Converter 并执行 INT8 PTQ."""
    if args.samples <= 0 or args.offset < 0:
        raise ValueError("samples 必须大于 0, offset 不能小于 0.")
    converter = mtk_converter.OnnxConverter.from_model_proto_file(
        str(args.onnx))
    converter.quantize = True
    converter.calibration_data_gen = lambda: calibration_data(
        args.image_dir, args.annotations, args.samples, args.offset)
    converter.use_per_output_channel_quantization = True
    converter.convert_to_tflite(str(args.output))


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    convert_model(parse_args())
