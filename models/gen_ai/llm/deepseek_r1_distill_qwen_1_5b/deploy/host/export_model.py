"""离线导出 Qwen2 静态解码分片,保留官方权重与 KV Cache."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn


def stable_rms_norm(hidden, norm):
    """先按幅度缩放再计算 RMSNorm,保持公式且避免 FP16 平方溢出."""
    scale = hidden.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
    normalized = hidden / scale
    epsilon = (norm.variance_epsilon ** 0.5 / scale).square()
    variance = normalized.square().mean(dim=-1, keepdim=True)
    return normalized * torch.rsqrt(variance + epsilon) * norm.weight


class DecoderLayer(nn.Module):
    """将官方解码层展开为静态矩阵运算,每次处理一个 Token."""

    def __init__(self, layer: nn.Module, heads: int, kv_heads: int):
        """复用官方参数,记录 GQA 头数."""
        super().__init__()
        self.layer = layer
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = layer.self_attn.q_proj.out_features // heads

    def forward(self, hidden, past_key, past_value, cosine, sine, mask):
        """执行 Attention 和 MLP,返回 Hidden 与当前 Token 的 KV."""
        norm = stable_rms_norm(hidden, self.layer.input_layernorm)
        attention = self.layer.self_attn
        query = attention.q_proj(norm).reshape(
            1, 1, self.heads, self.head_dim).transpose(1, 2)
        key = attention.k_proj(norm).reshape(
            1, 1, self.kv_heads, self.head_dim).transpose(1, 2)
        value = attention.v_proj(norm).reshape(
            1, 1, self.kv_heads, self.head_dim).transpose(1, 2)
        query_half = torch.cat(
            (-query[..., self.head_dim // 2:],
             query[..., :self.head_dim // 2]), dim=-1)
        key_half = torch.cat(
            (-key[..., self.head_dim // 2:],
             key[..., :self.head_dim // 2]), dim=-1)
        query = query * cosine + query_half * sine
        key = key * cosine + key_half * sine
        keys = torch.cat((past_key, key), dim=2)
        values = torch.cat((past_value, value), dim=2)
        groups = self.heads // self.kv_heads
        keys = torch.cat([keys[:, index:index + 1]
                          for index in range(self.kv_heads)
                          for _ in range(groups)], dim=1)
        values = torch.cat([values[:, index:index + 1]
                            for index in range(self.kv_heads)
                            for _ in range(groups)], dim=1)
        # 在矩阵乘法前缩放 Query,避免 FP16 中间点积溢出.
        scores = torch.matmul(query * self.head_dim ** -0.5,
                              keys.transpose(2, 3))
        probabilities = torch.softmax(scores + mask, dim=-1)
        context = torch.matmul(probabilities, values).transpose(1, 2)
        hidden = hidden + attention.o_proj(context.reshape(1, 1, -1))
        norm = stable_rms_norm(hidden, self.layer.post_attention_layernorm)
        hidden = hidden + self.layer.mlp(norm)
        return hidden, key, value


class OutputShard(nn.Module):
    """将官方输出投影按词表行分片,避免单个模型超过格式大小限制."""

    def __init__(self, norm, weight):
        """复用最终归一化和词表分片权重."""
        super().__init__()
        self.norm = norm
        self.register_buffer("weight", weight)

    def forward(self, hidden):
        """输出指定词表区间的 Logits."""
        return torch.nn.functional.linear(stable_rms_norm(hidden, self.norm),
                                          self.weight)


def sha256(path: Path) -> str:
    """分块计算文件哈希,不一次性读入大权重."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_graph(module, inputs, path, input_names, output_names):
    """导出固定形状 FP32 ONNX,保持命名 I/O 契约."""
    module.eval()
    torch.onnx.export(module, inputs, str(path), opset_version=13,
                      input_names=input_names, output_names=output_names,
                      do_constant_folding=True)


def export_model(args):
    """读取官方离线权重并按层导出,显示每个分片的进度."""
    from transformers import AutoModelForCausalLM

    args.output.mkdir(parents=True, exist_ok=True)
    record = json.loads((args.source / "source_manifest.json").read_text(encoding="utf-8-sig"))
    if record["revision"] != "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562":
        raise ValueError("官方资源 revision 不匹配.")
    for entry in record["files"]:
        print(f"[资源校验] {entry['name']}", flush=True)
        path = args.source / entry["name"]
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"官方资源哈希不匹配: {path}")
    shutil.copyfile(args.source / "source_manifest.json", args.output / "source_manifest.json")
    print("[导出] 加载离线官方权重.", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.source, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager").eval()
    config = model.config
    if config.model_type != "qwen2" or args.context < 2:
        raise ValueError("仅支持 Qwen2,上下文必须大于 1.")
    dim = config.hidden_size // config.num_attention_heads
    hidden = torch.zeros(1, 1, config.hidden_size)
    cache = torch.zeros(1, config.num_key_value_heads, args.context - 1, dim)
    angle = torch.zeros(1, 1, 1, dim)
    mask = torch.zeros(1, 1, 1, args.context)
    specs = []
    count = 1 if args.probe else len(model.model.layers)
    with torch.no_grad():
        for index, layer in enumerate(model.model.layers[:count]):
            name = f"layer_{index:02d}"
            print(f"[导出 {index + 1}/{count}] {name}", flush=True)
            export_graph(DecoderLayer(layer, config.num_attention_heads,
                                      config.num_key_value_heads),
                         (hidden, cache, cache, angle.cos(), angle.sin(), mask),
                         args.output / f"{name}.onnx",
                         ["hidden", "past_key", "past_value", "cosine",
                          "sine", "mask"], ["hidden_out", "key", "value"])
            specs.append({"name": name, "kind": "decoder"})
        if not args.probe:
            embedding = model.model.embed_tokens.weight.float().numpy()
            embedding.astype("<f2").tofile(args.output / "embedding_fp16.bin")
            for index, start in enumerate(range(0, config.vocab_size, 16384)):
                end = min(start + 16384, config.vocab_size)
                name = f"head_{index:02d}"
                print(f"[输出层] {name}: {start}:{end}", flush=True)
                export_graph(OutputShard(model.model.norm,
                                         model.lm_head.weight[start:end]),
                             hidden, args.output / f"{name}.onnx",
                             ["hidden"], ["logits"])
                specs.append({"name": name, "kind": "head",
                              "start": start, "count": end - start})
    manifest = {"context": args.context, "hidden": config.hidden_size,
                "implementation": "qwen2_static_stable_fp16_v1",
                "layers": count, "kv_heads": config.num_key_value_heads,
                "head_dim": dim, "vocab": config.vocab_size,
                "rope_theta": config.rope_theta, "precision": "fp32_onnx",
                "probe_only": args.probe, "graphs": specs}
    (args.output / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[导出] 静态 ONNX 分片已保存,尚未编译或板端验证.", flush=True)


def parse_args():
    """解析离线权重路径与候选上下文长度."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--probe", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    export_model(parse_args())
