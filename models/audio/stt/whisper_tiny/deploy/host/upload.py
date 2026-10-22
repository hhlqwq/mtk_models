"""上传 Whisper 双 DLA、Python 推理代码和原版文本组件."""

from pathlib import Path

from tools.deploy import arguments, upload


def main():
    """在运行板端示例前一并上传 tokenizer 与规范化资源."""
    ctx = arguments(__file__)
    vendor = ctx.work / "whisper_eval_vendor"
    files = [(str(Path("board") / path.relative_to(vendor).parent), path)
             for path in sorted((vendor / "whisper").rglob("*")) if path.is_file()]
    if not files:
        raise FileNotFoundError("缺少导出步骤准备的 Whisper 文本组件.")
    upload(ctx, [ctx.output / "encoder_fp32.dla", ctx.output / "decoder_step_fp32.dla",
                 ctx.work / "mel_filters_f32.bin", ctx.work / "decode_config.txt"],
           extra_files=files)


if __name__ == "__main__":
    main()
