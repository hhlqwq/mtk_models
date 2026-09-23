"""从锁定的 MobileFaceNet PyTorch 权重导出 ONNX。"""

import argparse
import hashlib
import sys
from pathlib import Path

import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "original" / "upstream"
sys.path.insert(0, str(SOURCE_ROOT))

from mobilefacenet import MobileFaceNet  # noqa: E402


def sha256(path: Path) -> str:
    """计算模型文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_model(weights: Path, output: Path, expected_hash: str) -> None:
    """验证原始权重并导出固定输入的特征提取图。"""
    actual_hash = sha256(weights)
    if actual_hash != expected_hash:
        raise ValueError(f"权重哈希不匹配: {actual_hash}")
    model = MobileFaceNet().eval()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    sample = torch.zeros((1, 3, 112, 112), dtype=torch.float32)
    with torch.no_grad():
        embedding = model(sample)
    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError(f"原始输出结构异常: {tuple(embedding.shape)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, sample, str(output), opset_version=13,
                      input_names=["face"], output_names=["embedding"],
                      do_constant_folding=True)
    print(f"[OK] PyTorch 输出: {tuple(embedding.shape)}")
    print(f"[OK] ONNX: {output}, SHA-256: {sha256(output)}")


def main() -> None:
    """解析参数并执行导出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    export_model(args.weights, args.output, args.sha256.lower())


if __name__ == "__main__":
    main()
