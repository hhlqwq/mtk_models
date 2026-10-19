"""读取板端配置执行 DeepSeek 固定样例,快速结果仅追加编号后缀."""

from tools.board import command, context


def main():
    """沿用原部署模型与桥接库,使用本次独立结果位置."""
    ctx = context(__file__)
    command(ctx, "evaluate.py", "--models", ctx.root / "models",
            "--library", ctx.root / "board/libneuron_bridge.so",
            "--samples", ctx.root / "models/samples.json",
            "--output", ctx.result, "--run-id", ctx.run_id)


if __name__ == "__main__":
    main()
