"""读取模型配置并使用 NCC 生成最终 DLA."""

from tools.deploy import arguments, compile_dla, python


def main():
    """执行本模型的编译步骤并打印最终 DLA 路径."""
    ctx = arguments(__file__)
    for name in ("encoder", "decoder_step"):
        compile_dla(ctx, ctx.work / f"{name}_fp32.tflite",
                    ctx.output / f"{name}_fp32.dla", ("--suppress-input",))


if __name__ == "__main__":
    main()
