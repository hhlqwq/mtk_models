"""读取本次 C++ 音频推理输出并计算精度,不执行硬件推理."""

from tools.board import command, context


def main():
    """保留当前模型的音频评测和示例输出协议."""
    ctx = context(__file__)
    cache = ctx.work / "cache/librispeech"
    command(ctx, "evaluate_accuracy.py", "--dataset", "librispeech",
            "--source-manifest", cache / "source_manifest.jsonl",
            "--board-predictions", ctx.work / "board_predictions.jsonl",
            "--preprocess-metrics", cache / "preprocess_metrics.jsonl",
            "--output-dir", ctx.work / "report")


if __name__ == "__main__":
    main()
