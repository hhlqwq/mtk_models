"""读取板端配置并调度 C++ 推理,不计算精度指标."""

from tools.board import command, context


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



if __name__ == "__main__":
    main()
