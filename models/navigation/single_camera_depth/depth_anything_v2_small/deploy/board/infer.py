"""读取板端 JSON 配置,完成本模型推理与原始指标计算."""

from tools.board import command, context
from tools.evaluation import is_test


def main():
    """执行当前运行,快速结果仅在 results 下追加 _test 后缀."""
    ctx = context(__file__)
    if ctx.result.exists():
        raise FileExistsError(f"结果目录已存在: {ctx.result}.")
    print(f"[板端] 结果目录: {ctx.result}", flush=True)
    command(ctx, "full_accuracy_board.py", "--dataset-root", ctx.dataset,
            "--model", ctx.root / "models/model_int8.dla",
            "--metadata", ctx.root / "models/model_int8.json",
            "--run-dir", ctx.work, "--run-id", ctx.run_id)
    inputs = sorted((ctx.work / "inputs").glob("*.bin"))
    if not inputs:
        raise FileNotFoundError("缺少当前推理输入.")
    command(ctx, "benchmark_board", "--model", ctx.root / "models/model_int8.dla",
            "--input", inputs[0], "--report", ctx.work / "report/benchmark.json",
            "--warmup", "1" if is_test() else "10", "--repeats", "5" if is_test() else "100")


if __name__ == "__main__":
    main()
