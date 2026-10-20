"""读取本次 C++ 输出并执行精度评测,不调用硬件推理."""

from tools.board import command, context


def main():
    """使用与推理阶段相同的配置和运行编号,保留原有指标协议."""
    ctx = context(__file__)
    command(ctx, "evaluate_full_accuracy.py", "--predictions", ctx.work / "predictions.jsonl",
            "--labels", ctx.dataset / "val_labels_0based.txt",
            "--report", ctx.work / "report", "--run-id", ctx.run_id)


if __name__ == "__main__":
    main()
