"""从官方 RTMPose 配置与 checkpoint 导出兼容 ONNX."""

import importlib.util
from pathlib import Path

from tools.deploy import arguments, python


def main():
    """保留官方导出接口,把项目 YAML 与 MMPose 模型定义明确区分."""
    ctx = arguments(__file__)
    module = importlib.util.find_spec("mmpose")
    if module is None:
        raise ImportError("当前容器缺少已锁定的 MMPose 环境.")
    config = Path(module.origin).parent / ".mim" / ctx.config["model"]["upstream_config"]
    if not config.is_file():
        raise FileNotFoundError(f"缺少官方 MMPose 模型定义: {config}.")
    python(ctx, "export_onnx.py", "--config", config,
           "--weights", ctx.config["model"]["weights"], "--output", ctx.output / "model_fp32.onnx")
    python(ctx, "prepare_onnx.py", "--source", ctx.output / "model_fp32.onnx",
           "--output", ctx.output / "model_mtk_compatible.onnx")


if __name__ == "__main__":
    main()
