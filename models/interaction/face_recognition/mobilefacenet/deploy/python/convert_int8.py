"""使用对齐人脸校准 MobileFaceNet 并生成 MTK INT8 TFLite。"""

import argparse
from collections.abc import Iterator
from pathlib import Path

import mtk_converter
import numpy as np

from face_utils import load_aligned_face


def calibration_data(images: list[Path]) -> Iterator[list[np.ndarray]]:
    """逐张提供真实对齐人脸输入并报告进度。"""
    for index, path in enumerate(images, start=1):
        print(f"[CALIBRATION] {index}/{len(images)} {path.name}", flush=True)
        yield [load_aligned_face(path)]


def convert_model(onnx: Path, image_dir: Path, output: Path) -> None:
    """对 ONNX 执行后训练量化。"""
    images = sorted(image_dir.glob("*_aligned.jpg"))
    if len(images) < 8:
        raise ValueError(f"校准人脸不足 8 张: {image_dir}")
    converter = mtk_converter.OnnxConverter.from_model_proto_file(str(onnx))
    converter.quantize = True
    converter.calibration_data_gen = lambda: calibration_data(images)
    converter.use_per_output_channel_quantization = True
    output.parent.mkdir(parents=True, exist_ok=True)
    converter.convert_to_tflite(str(output))
    print(f"[OK] INT8 TFLite: {output}")


def main() -> None:
    """解析量化参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert_model(args.onnx, args.image_dir, args.output)


if __name__ == "__main__":
    main()
