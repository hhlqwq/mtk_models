"""读取板端 JSON 配置,完成本模型推理与原始指标计算."""

from tools.board import command, context
from tools.evaluation import is_test


def main():
    """执行当前运行,快速结果仅在 results 下追加 _test 后缀."""
    ctx = context(__file__)
    if ctx.result.exists():
        raise FileExistsError(f"结果目录已存在: {ctx.result}.")
    print(f"[板端] 结果目录: {ctx.result}", flush=True)
    images = ctx.dataset / "val"
    if not images.is_dir():
        images = ctx.dataset / "ILSVRC2012_img_val"
    (ctx.work / "report").mkdir(parents=True)
    command(ctx, "vit_board_eval", "--model", ctx.root / "models/model_int8.dla",
            "--images", images, "--quantization", ctx.root / "models/quantization.json",
            "--predictions", ctx.work / "predictions.jsonl")
    command(ctx, "evaluate_full_accuracy.py", "--predictions", ctx.work / "predictions.jsonl",
            "--labels", ctx.dataset / "val_labels_0based.txt",
            "--report", ctx.work / "report", "--run-id", ctx.run_id)


if __name__ == "__main__":
    main()
