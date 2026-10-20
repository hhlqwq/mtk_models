"""读取模型配置并上传到原有板端部署目录."""

import os

from tools.deploy import arguments, upload
from tools.evaluation import is_test
from tools.process import run_command


def main():
    """上传模型、程序和 JSON 配置,保留全量与快速结果的统一部署位置."""
    ctx = arguments(__file__)
    upload(ctx, [
        ctx.output / "model_int8.dla",
], [
        ctx.work / "rtmpose_board_eval",
        ctx.work / "prepare_eval_manifest",
    ])


if __name__ == "__main__":
    main()
