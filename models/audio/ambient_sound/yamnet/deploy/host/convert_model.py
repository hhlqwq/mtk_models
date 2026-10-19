"""将 YAMNet 转换中间文件集中到 build."""

from tools.deploy import arguments, python


def main():
    """只读取既有校准参数,快速模式不缩减校准数据."""
    ctx = arguments(__file__)
    python(ctx, "convert_int8.py", "--models-dir", ctx.output, "--dataset", ctx.dataset,
           "--precision", ctx.config["conversion"]["precision"], "--work-dir", ctx.work,
           "--output-dir", ctx.work)


if __name__ == "__main__":
    main()
