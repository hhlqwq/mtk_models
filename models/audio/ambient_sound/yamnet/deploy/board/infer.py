"""读取板端 JSON 配置,完成本模型推理与原始指标计算."""

from tools.board import command, context
from tools.evaluation import is_test


def main():
    """执行当前运行,快速结果仅在 results 下追加 _test 后缀."""
    ctx = context(__file__)
    if ctx.result.exists():
        raise FileExistsError(f"结果目录已存在: {ctx.result}.")
    print(f"[板端] 结果目录: {ctx.result}", flush=True)
    precision = ctx.config["conversion"]["precision"]
    runtime = ctx.root / "models" / f"runtime_config_{precision}.csv"
    if not runtime.is_file() and precision in ("fp16", "int8"):
        runtime = ctx.root / "models/runtime_config.csv"
    command(ctx, "evaluate.py", "--mode", "prepare", "--dataset", ctx.dataset, "--work", ctx.work)
    command(ctx, "yamnet_eval", ctx.root / "models" / f"model_{precision}.dla", runtime, ctx.work)
    command(ctx, "evaluate.py", "--mode", "summarize", "--dataset", ctx.dataset, "--work", ctx.work,
            "--models-dir", ctx.root / "models", "--output", ctx.result,
            "--run-id", ctx.run_id, "--precision", precision)


if __name__ == "__main__":
    main()
