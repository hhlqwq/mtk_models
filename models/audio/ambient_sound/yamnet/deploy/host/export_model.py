"""读取 YAMNet 配置并执行当前独立浮点步骤."""

import os

from tools.deploy import arguments, python


def main():
    """TensorFlow 使用公共专用环境,路径由 Docker 启动时加载."""
    ctx = arguments(__file__)
    python(ctx, "export_evaluate.py", "--mode", "export", "--models-dir", ctx.output,
           "--dataset", ctx.dataset, executable=os.environ["MTK_YAMNET_PYTHON"])


if __name__ == "__main__":
    main()
