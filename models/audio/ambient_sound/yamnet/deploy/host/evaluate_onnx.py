"""读取 YAMNet 配置并执行当前独立浮点步骤."""

import os

from tools.deploy import arguments, python


def main():
    """TensorFlow 使用公共专用环境,路径由 Docker 启动时加载."""
    ctx = arguments(__file__)
    python(ctx, "export_evaluate.py", "--mode", "tensorflow", "--models-dir", ctx.output,
           "--dataset", ctx.dataset, "--output", ctx.work / "tensorflow",
           executable=os.environ["MTK_YAMNET_PYTHON"])
    python(ctx, "export_evaluate.py", "--mode", "onnx", "--models-dir", ctx.output,
           "--dataset", ctx.dataset, "--output", ctx.work / "onnx",
           "--reference", ctx.work / "tensorflow")


if __name__ == "__main__":
    main()
