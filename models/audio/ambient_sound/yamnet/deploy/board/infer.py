"""读取板端配置并调度 C++ 推理,不计算精度指标."""

from argparse import Namespace

from full_accuracy_board import prepare
from tools.board import command, context


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
    prepare(Namespace(dataset=ctx.dataset, work=ctx.work))
    command(ctx, "yamnet_eval", ctx.root / "models" / f"model_{precision}.dla", runtime, ctx.work)



if __name__ == "__main__":
    main()
