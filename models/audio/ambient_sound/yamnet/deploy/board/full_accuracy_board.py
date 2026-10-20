"""读取本次 C++ 音频推理输出并计算精度,不执行硬件推理."""

from tools.board import command, context


def main():
    """保留当前模型的音频评测和示例输出协议."""
    ctx = context(__file__)
    precision = ctx.config["conversion"]["precision"]
    command(ctx, "evaluate.py", "--mode", "summarize", "--dataset", ctx.dataset, "--work", ctx.work,
            "--models-dir", ctx.root / "models", "--output", ctx.result,
            "--run-id", ctx.run_id, "--precision", precision)


if __name__ == "__main__":
    main()
