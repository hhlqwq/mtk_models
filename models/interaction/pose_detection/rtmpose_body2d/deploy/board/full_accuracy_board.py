"""读取本次 C++ 输出并执行精度评测,不调用硬件推理."""

from tools.board import command, context


def main():
    """使用与推理阶段相同的配置和运行编号,保留原有指标协议."""
    ctx = context(__file__)
    command(ctx, "evaluate_coco_wholebody.py", "--annotations",
            ctx.dataset / "annotations/coco_wholebody_val_v1.0.json",
            "--predictions", ctx.work / "predictions.jsonl",
            "--formatted", ctx.work / "wholebody_predictions.json",
            "--metrics", ctx.work / "coco_wholebody_metrics.json",
            "--summary-log", ctx.work / "coco_wholebody_summary.log")


if __name__ == "__main__":
    main()
