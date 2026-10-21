"""读取板端配置并调度 C++ 推理,不计算精度指标."""

from tools.board import command, context
from tools.evaluation import is_test


def main():
    """执行当前运行,快速结果仅在 results 下追加 _test 后缀."""
    ctx = context(__file__)
    if ctx.result.exists():
        raise FileExistsError(f"结果目录已存在: {ctx.result}.")
    print(f"[板端] 结果目录: {ctx.result}", flush=True)
    ctx.result.mkdir(parents=True)
    arguments = ["--model", ctx.root / "models/model_int8.dla",
                 "--images", ctx.dataset / "images", "--output-dir", ctx.work,
                 "--warmup", "0" if is_test() else "20", "--confidence", "0.001", "--iou", "0.6", "--max-det", "300"]
    arguments += ["--config", ctx.root / "models/runtime_config.csv"]
    command(ctx, "yolov8n_board_eval", *arguments)



if __name__ == "__main__":
    main()
