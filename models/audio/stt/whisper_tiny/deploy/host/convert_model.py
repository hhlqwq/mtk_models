"""将 Whisper-Tiny FP32 ONNX 转换为 MTK FP32 TFLite."""

import argparse
from pathlib import Path

import mtk_converter

from tools.deploy import configure_parser


def convert_model(source: Path, output: Path) -> None:
    """使用 MTK Converter 转换单个静态 ONNX.

    Args:
        source: FP32 ONNX 路径.
        output: TFLite 输出路径.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    converter = mtk_converter.OnnxConverter.from_model_proto_file(str(source))
    converter.quantize = False
    converter.convert_to_tflite(str(output))


def main() -> None:
    """默认转换 Encoder 和 Decoder,显式参数仍支持单个 ONNX."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--output", type=Path)
    ctx = configure_parser(parser, __file__, {})
    args = parser.parse_args()
    if bool(args.onnx) != bool(args.output):
        parser.error("--onnx 和 --output 必须一起指定.")
    if args.onnx:
        convert_model(args.onnx, args.output)
    else:
        for index, name in enumerate(("encoder", "decoder_step"), 1):
            print(f"[转换 {index}/2] {name}", flush=True)
            convert_model(ctx.output / f"{name}_fp32.onnx", ctx.work / f"{name}_fp32.tflite")


if __name__ == "__main__":
    main()
