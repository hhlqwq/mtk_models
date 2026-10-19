"""读取板端 JSON 配置,完成本模型推理与原始指标计算."""

from tools.board import command, context
from tools.evaluation import is_test


def main():
    """执行当前运行,快速结果仅在 results 下追加 _test 后缀."""
    ctx = context(__file__)
    if ctx.result.exists():
        raise FileExistsError(f"结果目录已存在: {ctx.result}.")
    print(f"[板端] 结果目录: {ctx.result}", flush=True)
    cache = ctx.work / "cache/librispeech"
    (ctx.work / "report").mkdir(parents=True)
    command(ctx, "prepare_board_audio", "--dataset-root", ctx.dataset,
            "--filters", ctx.root / "models/mel_filters_f32.bin", "--output-dir", cache)
    command(ctx, "whisper_board_eval", ctx.root / "models/encoder_fp32.dla",
            ctx.root / "models/decoder_step_fp32.dla", cache / "board_manifest.tsv",
            ctx.root / "models/decode_config.txt", ctx.work / "board_predictions.jsonl")
    command(ctx, "evaluate_accuracy.py", "--dataset", "librispeech",
            "--source-manifest", cache / "source_manifest.jsonl",
            "--board-predictions", ctx.work / "board_predictions.jsonl",
            "--preprocess-metrics", cache / "preprocess_metrics.jsonl",
            "--output-dir", ctx.work / "report")


if __name__ == "__main__":
    main()
