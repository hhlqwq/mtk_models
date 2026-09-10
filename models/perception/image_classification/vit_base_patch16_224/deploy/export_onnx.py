"""从 TorchVision 官方 ViT-B/16 权重离线导出 ONNX."""

import argparse
import hashlib
from pathlib import Path

import onnx
import torch
from torch import nn
import torchvision
from torchvision.models import vit_b_16


EXPECTED_TORCHVISION_VERSION = "0.15.1"
EXPECTED_WEIGHTS_SHA256 = (
    "c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9")
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class NormalizedViT(nn.Module):
    """为 TorchVision ViT 增加固定的 ImageNet 输入归一化."""

    def __init__(self, model: nn.Module) -> None:
        """初始化导出包装器.

        Args:
            model: 已加载官方权重的 TorchVision ViT 模型.
        """
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """对 NCHW RGB `[0,1]` 输入归一化并输出 ImageNet logits."""
        return self.model((image - self.mean) / self.std)


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


def load_model(weights_path: Path, approximate_gelu: bool) -> nn.Module:
    """加载官方权重并构造导出模型.

    Args:
        weights_path: TorchVision 官方权重路径.
        approximate_gelu: 是否将 GELU 改为标准 tanh 近似.

    Returns:
        包含输入归一化的推理模型.
    """
    installed_version = torchvision.__version__.split("+")[0]
    if installed_version != EXPECTED_TORCHVISION_VERSION:
        raise RuntimeError(
            "TorchVision 版本不匹配: "
            f"expected={EXPECTED_TORCHVISION_VERSION}, "
            f"actual={installed_version}")
    actual_sha256 = sha256_file(weights_path)
    if actual_sha256 != EXPECTED_WEIGHTS_SHA256:
        raise ValueError(
            f"ViT 权重 SHA-256 不匹配: {actual_sha256}")

    model = vit_b_16(weights=None)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    if approximate_gelu:
        for module in model.modules():
            if isinstance(module, nn.GELU):
                module.approximate = "tanh"
    model.eval()
    return NormalizedViT(model).eval()


def export_model(
        weights_path: Path,
        output_path: Path,
        approximate_gelu: bool) -> None:
    """导出固定输入 ViT ONNX 并执行结构检查.

    Args:
        weights_path: TorchVision 官方权重路径.
        output_path: ONNX 输出路径.
        approximate_gelu: 是否使用 tanh GELU.
    """
    model = load_model(weights_path, approximate_gelu)
    dummy_input = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
    )
    exported = onnx.load(str(output_path))
    onnx.checker.check_model(exported)
    print(f"[OK] ViT ONNX 已导出: {output_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数.

    Returns:
        已解析参数.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--compatible-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """依次导出精确基线和 MTK 兼容候选模型."""
    args = parse_args()
    print("[1/2] 导出官方精确 GELU 基线 ONNX.")
    export_model(args.weights, args.reference_output, False)
    print("[2/2] 导出 tanh GELU 的 MTK 兼容候选 ONNX.")
    export_model(args.weights, args.compatible_output, True)


if __name__ == "__main__":
    main()
