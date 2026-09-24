"""从固定的官方 Depth Anything V2 Small 权重导出静态 ONNX。"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import torch


SOURCE_REVISION = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
WEIGHTS_SHA256 = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"


class ExportableAttention(torch.nn.Module):
    """将官方 QKV 权重用于最多四维的固定自注意力计算。"""

    def __init__(self, original: torch.nn.Module) -> None:
        """复用官方注意力层参数，不修改权重内容。"""
        super().__init__()
        self.qkv = original.qkv
        self.proj = original.proj
        self.num_heads = original.num_heads
        self.scale = original.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """用四维 Q、K、V 张量执行多头自注意力。"""
        batch, tokens, channels = x.shape
        projected = self.qkv(x)
        query, key, value = projected.chunk(3, dim=-1)
        head_size = channels // self.num_heads
        query = query.reshape(batch, tokens, self.num_heads, head_size)
        key = key.reshape(batch, tokens, self.num_heads, head_size)
        value = value.reshape(batch, tokens, self.num_heads, head_size)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attended = torch.matmul(torch.softmax(scores, dim=-1), value)
        merged = attended.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(merged)


def sha256_file(path: Path) -> str:
    """逐块计算文件哈希，避免将大权重一次读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_model(upstream: Path, weights: Path, output: Path, size: int) -> None:
    """核验官方源码和权重，导出固定方形输入的深度图。"""
    if size <= 0 or size % 14:
        raise ValueError("输入边长必须是 14 的正整数倍。")
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={upstream.resolve()}",
         "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"官方源码提交不匹配: {revision}")
    actual_hash = sha256_file(weights)
    if actual_hash != WEIGHTS_SHA256:
        raise ValueError(f"官方权重 SHA-256 不匹配: {actual_hash}")

    sys.path.insert(0, str(upstream))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.rand((1, 3, size, size),
                        generator=torch.Generator().manual_seed(20260924))
    patch_count = (size // 14) ** 2
    token = torch.zeros((1, patch_count + 1, model.pretrained.embed_dim))
    with torch.no_grad():
        baseline = model(sample)
        position = model.pretrained.interpolate_pos_encoding(token, size, size)
        model.pretrained.pos_embed = torch.nn.Parameter(position)
        for block in model.pretrained.blocks:
            block.attn = ExportableAttention(block.attn)
        revised = model(sample)
    max_difference = (baseline - revised).abs().max().item()
    if max_difference > 1e-4:
        raise ValueError(f"固定位置编码与原模型输出不一致: {max_difference}")
    print(f"[EXPORT] 固定位置编码和四维注意力，PyTorch 最大差异 {max_difference:.8g}。",
          flush=True)
    print(f"[EXPORT] 固定输入 1x3x{size}x{size}，输出相对深度图。", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            model, sample, str(output), opset_version=17,
            input_names=["image"], output_names=["relative_depth"],
            do_constant_folding=True)
    print(f"[OK] ONNX: {output}; SHA-256: {sha256_file(output)}", flush=True)


def main() -> None:
    """读取命令行参数并执行导出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=518)
    args = parser.parse_args()
    export_model(args.upstream, args.weights, args.output, args.size)


if __name__ == "__main__":
    main()
