"""读取模型配置并上传到原有板端部署目录."""

from tools.deploy import arguments, upload


def main():
    """上传模型、程序和 JSON 配置,保留全量与快速结果的统一部署位置."""
    ctx = arguments(__file__)
    upload(ctx, [
        ctx.output / "model_int8.dla",
        ctx.work / "board_input/metadata.json",
], [
        ctx.work / "mobilefacenet_board",
    ])


if __name__ == "__main__":
    main()
