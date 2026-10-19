"""读取 YAML 并执行与板端同协议的浮点精度评测."""

import sys

from tools.deploy import arguments, python, reference_path
from tools.evaluation import is_test
from tools.process import run_command


def main():
    """评测当前模式的样本,快速参考单独保存."""
    ctx = arguments(__file__)
    for backend, flag, path in (
            ("pytorch", "--weights", ctx.config["model"]["weights"]),
            ("onnx", "--onnx", ctx.output / "model_fp32.onnx")):
        output = reference_path(ctx) if backend == "onnx" else ctx.work / (
            "pytorch_reference_test.json" if is_test() else "pytorch_reference.json")
        python(ctx, "evaluate_fp32.py", flag, path,
               "--dataset-root", ctx.dataset, "--output", output)


if __name__ == "__main__":
    main()
