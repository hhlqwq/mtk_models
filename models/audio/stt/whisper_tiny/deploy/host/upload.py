"""读取模型配置并上传到原有板端部署目录."""

import os
import shutil

from tools.deploy import arguments, python, upload
from tools.evaluation import is_test
from tools.process import run_command


def main():
    """上传模型、程序和 JSON 配置,保留全量与快速结果的统一部署位置."""
    ctx = arguments(__file__)
    python(ctx, "export_board_assets.py", "--output-dir", ctx.work)
    python(ctx, "export_eval_vendor.py", "--output-dir", ctx.work / "whisper_eval_vendor")
    upload(ctx, [
        ctx.output / "encoder_fp32.dla",
        ctx.output / "decoder_step_fp32.dla",
        ctx.work / "mel_filters_f32.bin",
        ctx.work / "decode_config.txt",
], [
        ctx.work / "prepare_board_audio",
        ctx.work / "whisper_board_eval",
    ])
    run_command(["scp", "-r", ctx.work / "whisper_eval_vendor/whisper",
                 f"{os.environ['MTK_BOARD_HOST']}:{ctx.config['board']['deploy_dir']}/board/"])


if __name__ == "__main__":
    main()
