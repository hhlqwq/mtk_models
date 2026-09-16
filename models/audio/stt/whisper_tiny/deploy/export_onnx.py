"""从 OpenAI 官方 tiny.pt 导出固定 Shape Encoder 与 Decoder-Step ONNX."""

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn


MAX_DECODE_TOKENS = 200
MASK_NEGATIVE = -100.0


class EncoderWrapper(nn.Module):
    """封装 Whisper Encoder,固定返回音频特征."""

    def __init__(self, encoder: nn.Module) -> None:
        """初始化 Encoder 包装器.

        Args:
            encoder: OpenAI Whisper Encoder 模块.
        """
        super().__init__()
        self.encoder = encoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """运行固定 30 秒 Log-Mel Encoder.

        Args:
            mel: `[1, 80, 3000]` FP32 Log-Mel.

        Returns:
            `[1, 1500, n_audio_state]` 音频特征.
        """
        return self.encoder(mel)


class DecoderStepWrapper(nn.Module):
    """将 OpenAI Decoder 改写为固定 KV Cache 的单 Token 图."""

    def __init__(self, decoder: nn.Module, max_tokens: int) -> None:
        """初始化 Decoder-Step 包装器.

        Args:
            decoder: OpenAI Whisper TextDecoder 模块.
            max_tokens: 静态 KV Cache Token 上限.
        """
        super().__init__()
        self.decoder = decoder
        self.max_tokens = max_tokens

    @staticmethod
    def _project_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
        """将 `[B, T, C]` 投影结果拆为多头布局.

        Args:
            tensor: 注意力投影张量.
            heads: 注意力头数.

        Returns:
            `[B, H, T, D]` 张量.
        """
        batch, tokens, channels = tensor.shape
        return tensor.reshape(batch, tokens, heads, channels // heads).permute(
            0, 2, 1, 3)

    def _self_attention(
            self, attention: nn.Module, hidden: torch.Tensor,
            past_key: torch.Tensor, past_value: torch.Tensor,
            cache_update_mask: torch.Tensor,
            attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor]:
        """执行带固定长度 KV Cache 的自注意力.

        Args:
            attention: OpenAI MultiHeadAttention 模块.
            hidden: 当前 Token 隐状态.
            past_key: 固定长度历史 Key Cache.
            past_value: 固定长度历史 Value Cache.
            cache_update_mask: 当前写入位置的 One-Hot Mask.
            attention_mask: 屏蔽未写入 Cache 的加法 Mask.

        Returns:
            注意力输出、更新后的 Key Cache 和 Value Cache.
        """
        heads = attention.n_head
        query = self._project_heads(attention.query(hidden), heads)
        key_new = self._project_heads(attention.key(hidden), heads)
        value_new = self._project_heads(attention.value(hidden), heads)
        update = cache_update_mask.reshape(1, 1, self.max_tokens, 1)
        key = past_key * (1.0 - update) + key_new * update
        value = past_value * (1.0 - update) + value_new * update
        scale = (query.shape[-1]) ** -0.25
        scores = (query * scale) @ (key * scale).transpose(-1, -2)
        scores = scores + attention_mask
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        context = probabilities @ value
        context = context.permute(0, 2, 1, 3).flatten(start_dim=2)
        return attention.out(context), key, value

    def _cross_attention(
            self, attention: nn.Module, hidden: torch.Tensor,
            audio_features: torch.Tensor) -> torch.Tensor:
        """执行当前 Token 到 Encoder 特征的交叉注意力.

        Args:
            attention: OpenAI MultiHeadAttention 模块.
            hidden: 当前 Token 隐状态.
            audio_features: Encoder 输出.

        Returns:
            交叉注意力输出.
        """
        heads = attention.n_head
        query = self._project_heads(attention.query(hidden), heads)
        key = self._project_heads(attention.key(audio_features), heads)
        value = self._project_heads(attention.value(audio_features), heads)
        scale = (query.shape[-1]) ** -0.25
        scores = (query * scale) @ (key * scale).transpose(-1, -2)
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        context = probabilities @ value
        context = context.permute(0, 2, 1, 3).flatten(start_dim=2)
        return attention.out(context)

    def forward(
            self, token: torch.Tensor, audio_features: torch.Tensor,
            kv_cache: torch.Tensor, position_weights: torch.Tensor,
            cache_update_mask: torch.Tensor,
            attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """运行一个 Decoder Token 并返回 Logits 和完整新 Cache.

        Args:
            token: `[1, 1]` INT64 Token.
            audio_features: `[1, 1500, n_audio_state]` Encoder 输出.
            kv_cache: `[L, 2, 1, H, M, D]` 固定 Cache.
            position_weights: `[1, n_text_ctx]` 位置 One-Hot 权重.
            cache_update_mask: `[1, M]` Cache 写入 One-Hot Mask.
            attention_mask: `[1, 1, 1, M]` 加法注意力 Mask.

        Returns:
            `[1, vocab]` Logits 与更新后的固定 KV Cache.
        """
        position = position_weights @ self.decoder.positional_embedding
        hidden = self.decoder.token_embedding(token) + position.unsqueeze(1)
        next_layers = []
        for index, block in enumerate(self.decoder.blocks):
            residual = hidden
            attention_output, key, value = self._self_attention(
                block.attn, block.attn_ln(hidden), kv_cache[index, 0],
                kv_cache[index, 1], cache_update_mask, attention_mask)
            hidden = residual + attention_output
            residual = hidden
            hidden = residual + self._cross_attention(
                block.cross_attn, block.cross_attn_ln(hidden), audio_features)
            hidden = hidden + block.mlp(block.mlp_ln(hidden))
            next_layers.append(torch.stack((key, value), dim=0))
        hidden = self.decoder.ln(hidden)
        logits = hidden @ self.decoder.token_embedding.weight.transpose(0, 1)
        return logits[:, 0, :].float(), torch.stack(next_layers, dim=0)


def load_model(weights: Path) -> Any:
    """从已校验的本地 checkpoint 加载 OpenAI Whisper.

    Args:
        weights: `tiny.pt` 路径.

    Returns:
        CPU FP32 Eval 模式 Whisper 模型.
    """
    import whisper

    model = whisper.load_model(str(weights), device="cpu")
    return model.float().eval()


def export_models(weights: Path, output_dir: Path,
                  max_tokens: int) -> None:
    """导出 Encoder 与 Decoder-Step ONNX.

    Args:
        weights: OpenAI 官方权重路径.
        output_dir: ONNX 输出目录.
        max_tokens: 静态 KV Cache 长度.
    """
    if max_tokens <= 0 or max_tokens > 448:
        raise ValueError("max_tokens 必须位于 1..448.")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[1/3] 加载 OpenAI Whisper-Tiny FP32 模型.")
    model = load_model(weights)
    dimensions = model.dims
    encoder = EncoderWrapper(model.encoder).eval()
    decoder = DecoderStepWrapper(model.decoder, max_tokens).eval()
    mel = torch.zeros(1, dimensions.n_mels, 3000, dtype=torch.float32)

    print("[2/3] 导出固定 30 秒 Encoder ONNX.")
    encoder_path = output_dir / "encoder_fp32.onnx"
    torch.onnx.export(
        encoder, mel, str(encoder_path), input_names=["mel"],
        output_names=["audio_features"], opset_version=17,
        do_constant_folding=True)

    print("[3/3] 导出固定 KV Cache Decoder-Step ONNX.")
    with torch.no_grad():
        audio_features = encoder(mel)
    head_dim = dimensions.n_text_state // dimensions.n_text_head
    cache = torch.zeros(
        dimensions.n_text_layer, 2, 1, dimensions.n_text_head, max_tokens,
        head_dim, dtype=torch.float32)
    token = torch.zeros(1, 1, dtype=torch.int64)
    position_weights = torch.zeros(
        1, dimensions.n_text_ctx, dtype=torch.float32)
    position_weights[0, 0] = 1.0
    cache_update_mask = torch.zeros(1, max_tokens, dtype=torch.float32)
    cache_update_mask[0, 0] = 1.0
    attention_mask = torch.full(
        (1, 1, 1, max_tokens), MASK_NEGATIVE, dtype=torch.float32)
    attention_mask[..., 0] = 0.0
    decoder_path = output_dir / "decoder_step_fp32.onnx"
    torch.onnx.export(
        decoder,
        (token, audio_features, cache, position_weights, cache_update_mask,
         attention_mask),
        str(decoder_path),
        input_names=[
            "token", "audio_features", "kv_cache", "position_weights",
            "cache_update_mask", "attention_mask"
        ],
        output_names=["logits", "kv_cache_out"],
        opset_version=17,
        do_constant_folding=True)
    print(f"[OK] Encoder: {encoder_path}")
    print(f"[OK] Decoder-Step: {decoder_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数.

    Returns:
        命令行参数.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=MAX_DECODE_TOKENS)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    export_models(arguments.weights, arguments.output_dir,
                  arguments.max_tokens)
