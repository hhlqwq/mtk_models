"""验证 OpenAI、改写后的 Decoder-Step 与 ONNX Runtime 数值一致性."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from export_onnx import DecoderStepWrapper, EncoderWrapper, MASK_NEGATIVE
from export_onnx import load_model


def error_statistics(reference: np.ndarray,
                     actual: np.ndarray) -> dict[str, float]:
    """计算两个数组的绝对误差和余弦相似度.

    Args:
        reference: 参考数组.
        actual: 被比较数组.

    Returns:
        最大值、均值、P99 绝对误差和余弦相似度.
    """
    reference_fp64 = reference.astype(np.float64).reshape(-1)
    actual_fp64 = actual.astype(np.float64).reshape(-1)
    absolute = np.abs(reference_fp64 - actual_fp64)
    denominator = np.linalg.norm(reference_fp64) * np.linalg.norm(actual_fp64)
    cosine = float(np.dot(reference_fp64, actual_fp64) / denominator)
    return {
        "max": float(np.max(absolute)),
        "mean": float(np.mean(absolute)),
        "p99": float(np.percentile(absolute, 99)),
        "cosine": cosine,
    }


def format_statistics(statistics: dict[str, float]) -> str:
    """格式化数值对齐统计.

    Args:
        statistics: `error_statistics()` 返回值.

    Returns:
        单行可读摘要.
    """
    return (
        f"max={statistics['max']:.8g}, mean={statistics['mean']:.8g}, "
        f"p99={statistics['p99']:.8g}, cosine={statistics['cosine']:.10f}")


def run_verification(args: argparse.Namespace) -> None:
    """执行 Encoder 和多步 Decoder 的三方数值对齐.

    Args:
        args: 命令行参数.
    """
    torch.manual_seed(20260915)
    model = load_model(args.weights)
    dimensions = model.dims
    encoder = EncoderWrapper(model.encoder).eval()
    decoder = DecoderStepWrapper(model.decoder, args.max_tokens).eval()
    mel = torch.randn(1, dimensions.n_mels, 3000, dtype=torch.float32) * 0.1
    with torch.no_grad():
        encoder_reference = encoder(mel).numpy()
    encoder_session = ort.InferenceSession(
        str(args.encoder), providers=["CPUExecutionProvider"])
    encoder_onnx = encoder_session.run(None, {"mel": mel.numpy()})[0]
    encoder_statistics = error_statistics(encoder_reference, encoder_onnx)
    print(f"[CHECK] Encoder: {format_statistics(encoder_statistics)}")
    if (encoder_statistics["max"] > args.encoder_max_error or
            encoder_statistics["mean"] > args.encoder_mean_error or
            encoder_statistics["cosine"] < args.encoder_min_cosine):
        raise ValueError("Encoder ONNX 数值不一致.")

    decoder_session = ort.InferenceSession(
        str(args.decoder), providers=["CPUExecutionProvider"])
    head_dim = dimensions.n_text_state // dimensions.n_text_head
    kv_caches = []
    for _ in range(dimensions.n_text_layer):
        kv_caches.extend((
            np.zeros((dimensions.n_text_head, 1, head_dim, args.max_tokens),
                     dtype=np.float32),
            np.zeros((dimensions.n_text_head, 1, head_dim, args.max_tokens),
                     dtype=np.float32),
        ))
    token_ids = [50258, 50259, 50359, 50363]
    history: list[int] = []
    for position, token_id in enumerate(token_ids):
        history.append(token_id)
        token_onehot = np.zeros((1, dimensions.n_vocab), dtype=np.float32)
        token_onehot[0, token_id] = 1.0
        position_weights = np.zeros(
            (1, dimensions.n_text_ctx), dtype=np.float32)
        position_weights[0, position] = 1.0
        cache_update = np.zeros(
            (1, 1, 1, args.max_tokens), dtype=np.float32)
        cache_update[..., position] = 1.0
        attention_mask = np.full(
            (1, 1, 1, args.max_tokens), MASK_NEGATIVE, dtype=np.float32)
        attention_mask[..., :position + 1] = 0.0
        with torch.no_grad():
            original_logits = model.decoder(
                torch.tensor([history]),
                torch.from_numpy(encoder_reference))[:, -1, :].float().numpy()
            step_outputs = decoder(
                torch.from_numpy(token_onehot),
                torch.from_numpy(encoder_reference),
                torch.from_numpy(position_weights),
                torch.from_numpy(cache_update), torch.from_numpy(attention_mask),
                *[torch.from_numpy(cache) for cache in kv_caches])
        onnx_inputs = {
            "token_onehot": token_onehot,
            "audio_features": encoder_onnx,
            "position_weights": position_weights,
            "cache_update_mask": cache_update,
            "attention_mask": attention_mask,
        }
        for layer in range(dimensions.n_text_layer):
            onnx_inputs[f"key_cache_{layer}_in"] = kv_caches[layer * 2]
            onnx_inputs[f"value_cache_{layer}_in"] = kv_caches[layer * 2 + 1]
        onnx_outputs = decoder_session.run(None, onnx_inputs)
        step_logits = step_outputs[0]
        step_caches = step_outputs[1:]
        onnx_logits = onnx_outputs[0]
        onnx_caches = onnx_outputs[1:]
        original_statistics = error_statistics(
            original_logits, step_logits.numpy())
        onnx_statistics = error_statistics(step_logits.numpy(), onnx_logits)
        cache_max = max(
            error_statistics(reference.numpy(), actual)["max"]
            for reference, actual in zip(step_caches, onnx_caches))
        print(
            f"[CHECK] Decoder {position + 1}/{len(token_ids)}: "
            f"原始/改写 {format_statistics(original_statistics)}; "
            f"改写/ONNX {format_statistics(onnx_statistics)}; "
            f"Cache max={cache_max:.8g}")
        if original_statistics["max"] > args.decoder_max_error:
            raise ValueError(f"Decoder-Step 第 {position} 步与原始模型不一致.")
        if onnx_statistics["max"] > args.decoder_max_error:
            raise ValueError(f"Decoder-Step 第 {position} 步 ONNX 不一致.")
        if cache_max > args.cache_max_error:
            raise ValueError(f"Decoder-Step 第 {position} 步 Cache 不一致.")
        kv_caches = list(onnx_caches)
    print("[OK] OpenAI、Decoder-Step 与 ONNX Runtime 数值一致.")


def parse_args() -> argparse.Namespace:
    """解析命令行参数.

    Returns:
        命令行参数.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--encoder-max-error", type=float, default=5e-3)
    parser.add_argument("--encoder-mean-error", type=float, default=2e-4)
    parser.add_argument("--encoder-min-cosine", type=float, default=0.99999)
    parser.add_argument("--decoder-max-error", type=float, default=2e-4)
    parser.add_argument("--cache-max-error", type=float, default=2e-5)
    return parser.parse_args()


if __name__ == "__main__":
    run_verification(parse_args())
