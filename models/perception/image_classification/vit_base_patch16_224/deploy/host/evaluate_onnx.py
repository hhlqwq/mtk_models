"""读取 YAML 并执行与板端同协议的浮点精度评测."""

import sys

from tools.deploy import arguments, python, reference_path
from tools.evaluation import is_test
from tools.process import run_command


def main():
    """评测当前模式的样本,快速参考单独保存."""
    ctx = arguments(__file__)
    run_command([sys.executable, ctx.repo / "tools/accuracy/evaluate_onnx.py",
                 "--model", "vit_base_patch16_224", "--onnx", ctx.output / "model_mtk_compatible.onnx",
                 "--dataset-root", ctx.dataset, "--output", reference_path(ctx)])


if __name__ == "__main__":
    main()
