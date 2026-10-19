"""读取模型配置并上传到原有板端部署目录."""

import os

from tools.deploy import arguments, python, upload
from tools.evaluation import is_test
from tools.process import run_command


def main():
    """上传模型、程序和 JSON 配置,保留全量与快速结果的统一部署位置."""
    ctx = arguments(__file__)
    precision = ctx.config["conversion"]["precision"]
    python(ctx, "prepare_reference.py", "--model-root", ctx.model, "--models-dir", ctx.output,
           "--dataset", ctx.dataset, "--work", ctx.work)
    upload(ctx, [
        ctx.output / f"model_{precision}.dla",
        ctx.work / f"runtime_config_{precision}.csv",
        ctx.output / "yamnet_class_map.csv",
        ctx.output / ("fp32_reference_test.json" if is_test() else "fp32_reference.json"),
], [
        ctx.work / "yamnet_eval",
    ])


if __name__ == "__main__":
    main()
