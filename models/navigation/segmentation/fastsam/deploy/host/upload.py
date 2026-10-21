"""部署已有 FastSAM 模型并生成展示示例,不依赖精度参考."""

from tools.deploy import arguments, upload


def main():
    """复用公共打包入口,由板端 compile.sh 完成少量通路检查."""
    ctx = arguments(__file__)
    upload(ctx, [ctx.output / "model_int8.dla", ctx.work / "runtime_config.csv"],
           [ctx.work / "fastsam_board"])


if __name__ == "__main__":
    main()
