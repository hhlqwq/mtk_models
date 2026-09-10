"""从 OpenMMLab MMPose 官方 RTMPose-M 权重离线导出 ONNX."""

import argparse
import hashlib
import importlib.machinery
from pathlib import Path
import sys
import types

import mmpose
import onnx
import torch
from torch import nn


class MissingMmcvOps(types.ModuleType):
    """为未使用的 MMCV 扩展符号提供失败即停的导入占位."""

    def __getattr__(self, name: str):
        """返回一旦调用就明确失败的扩展函数.

        Args:
            name: MMCV 扩展函数名称.

        Returns:
            不可调用的占位函数.
        """
        def unavailable(*args, **kwargs):
            """阻止 RTMPose 导出静默调用未安装的 MMCV 扩展."""
            del args, kwargs
            raise RuntimeError(
                f"RTMPose 导出意外调用未安装的 MMCV 扩展: {name}")

        return unavailable


def prepare_mmcv_lite_import() -> None:
    """允许 MMPose 注册无关模型,但禁止实际调用缺失的 MMCV 扩展."""
    try:
        __import__("mmcv._ext")
    except ModuleNotFoundError:
        stub = MissingMmcvOps("mmcv._ext")
        stub.__file__ = "/virtual/mmcv/_ext.py"
        stub.__spec__ = importlib.machinery.ModuleSpec(
            "mmcv._ext", loader=None)
        sys.modules["mmcv._ext"] = stub


prepare_mmcv_lite_import()

from mmpose.apis import init_model


EXPECTED_MMPOSE_VERSION = "1.3.2"
EXPECTED_WEIGHTS_SHA256 = (
    "3da02694cd6479d3b333ff42ebd0723f96bfa06adac1db1e2e815ed2e9e1b02d")
INPUT_HEIGHT = 256
INPUT_WIDTH = 192
KEYPOINTS = 133
IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)


class RTMPoseExportWrapper(nn.Module):
    """导出 RTMPose 主干和 SimCC 头,并固定 RGB 输入归一化."""

    def __init__(self, model: nn.Module) -> None:
        """初始化导出包装器.

        Args:
            model: 由 MMPose 官方配置和权重构造的模型.
        """
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(
            self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """对 NCHW RGB `[0,255]` 输入归一化并输出两个 SimCC 张量."""
        normalized = (image - self.mean) / self.std
        features = self.model.extract_feat(normalized)
        pred_x, pred_y = self.model.head.forward(features)
        return pred_x, pred_y


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256.

    Args:
        path: 待计算文件.

    Returns:
        小写十六进制 SHA-256.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def export_model(config_path: Path, weights_path: Path, output_path: Path) -> None:
    """加载官方 MMPose 模型并导出固定输入 ONNX.

    Args:
        config_path: MMPose v1.3.2 官方配置路径.
        weights_path: 官方 RTMPose-M checkpoint 路径.
        output_path: ONNX 输出路径.
    """
    if mmpose.__version__ != EXPECTED_MMPOSE_VERSION:
        raise RuntimeError(
            "MMPose 版本不匹配: "
            f"expected={EXPECTED_MMPOSE_VERSION}, "
            f"actual={mmpose.__version__}")
    actual_sha256 = sha256_file(weights_path)
    if actual_sha256 != EXPECTED_WEIGHTS_SHA256:
        raise ValueError(
            f"RTMPose 权重 SHA-256 不匹配: {actual_sha256}")

    model = init_model(str(config_path), str(weights_path), device="cpu")
    model.eval()
    out_channels = int(model.head.out_channels)
    if out_channels != KEYPOINTS:
        raise ValueError(
            f"RTMPose 关键点数量错误: expected={KEYPOINTS}, "
            f"actual={out_channels}")
    wrapper = RTMPoseExportWrapper(model).eval()
    dummy_input = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["pred_x", "pred_y"],
    )
    exported = onnx.load(str(output_path))
    onnx.checker.check_model(exported)
    print(f"[OK] RTMPose ONNX 已导出: {output_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数.

    Returns:
        已解析参数.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    export_model(arguments.config, arguments.weights, arguments.output)
