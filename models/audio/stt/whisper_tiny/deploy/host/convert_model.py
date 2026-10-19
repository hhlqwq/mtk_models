"""将 Whisper 的 Encoder 和 Decoder ONNX 分别转换为 FP32 TFLite."""

from tools.deploy import arguments, python


def main():
    """读取配置,将两个中间模型保存到 build."""
    ctx = arguments(__file__)
    for name in ("encoder", "decoder_step"):
        python(ctx, "convert_fp32.py", "--onnx", ctx.output / f"{name}_fp32.onnx",
               "--output", ctx.work / f"{name}_fp32.tflite")


if __name__ == "__main__":
    main()
