"""将 Whisper-Tiny FP32 ONNX 转换为 MTK FP32 TFLite."""

import argparse
from pathlib import Path

import mtk_converter


def convert_model(source: Path, output: Path) -> None:
    """使用 MTK Converter 转换单个静态 ONNX.

    Args:
        source: FP32 ONNX 路径.
        output: TFLite 输出路径.
    """
    converter = mtk_converter.OnnxConverter.from_model_proto_file(str(source))
    converter.quantize = False
    converter.convert_to_tflite(str(output))


def parse_args() -> argparse.Namespace:
    """解析命令行参数.

    Returns:
        命令行参数.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    convert_model(arguments.onnx, arguments.output)
