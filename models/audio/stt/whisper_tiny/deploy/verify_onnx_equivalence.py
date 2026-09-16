"""验证 OpenAI、改写后的 Decoder-Step 与 ONNX Runtime 数值一致性."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from export_onnx import DecoderStepWrapper, EncoderWrapper, MASK_NEGATIVE
from export_onnx import load_model


def maximum_error(reference: np.ndarray, actual: np.ndarray) -> float:
    """计算两个数组的最大绝对误差.

    Args:
        reference: 参考数组.
        actual: 被比较数组.

    Returns:
        最大绝对误差.
    """
    return float(np.max(np.abs(reference.astype(np.float64) - actual)))


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
    encoder_error = maximum_error(encoder_reference, encoder_onnx)
    print(f"[CHECK] Encoder 最大绝对误差: {encoder_error:.8g}")
    if not np.allclose(encoder_reference, encoder_onnx,
                       atol=args.atol, rtol=args.rtol):
        raise ValueError("Encoder ONNX 数值不一致.")

    decoder_session = ort.InferenceSession(
        str(args.decoder), providers=["CPUExecutionProvider"])
    head_dim = dimensions.n_text_state // dimensions.n_text_head
    cache = np.zeros((
        dimensions.n_text_layer, 2, 1, dimensions.n_text_head,
        args.max_tokens, head_dim), dtype=np.float32)
    token_ids = [50258, 50259, 50359, 50363]
    history: list[int] = []
    for position, token_id in enumerate(token_ids):
        history.append(token_id)
        token = np.asarray([[token_id]], dtype=np.int64)
        position_weights = np.zeros(
            (1, dimensions.n_text_ctx), dtype=np.float32)
        position_weights[0, position] = 1.0
        cache_update = np.zeros((1, args.max_tokens), dtype=np.float32)
        cache_update[0, position] = 1.0
        attention_mask = np.full(
            (1, 1, 1, args.max_tokens), MASK_NEGATIVE, dtype=np.float32)
        attention_mask[..., :position + 1] = 0.0
        with torch.no_grad():
            original_logits = model.decoder(
                torch.tensor([history]),
                torch.from_numpy(encoder_reference))[:, -1, :].float().numpy()
            step_logits, step_cache = decoder(
                torch.from_numpy(token), torch.from_numpy(encoder_reference),
                torch.from_numpy(cache), torch.from_numpy(position_weights),
                torch.from_numpy(cache_update), torch.from_numpy(attention_mask))
        onnx_logits, onnx_cache = decoder_session.run(None, {
            "token": token,
            "audio_features": encoder_onnx,
            "kv_cache": cache,
            "position_weights": position_weights,
            "cache_update_mask": cache_update,
            "attention_mask": attention_mask,
        })
        original_error = maximum_error(original_logits, step_logits.numpy())
        onnx_error = maximum_error(step_logits.numpy(), onnx_logits)
        cache_error = maximum_error(step_cache.numpy(), onnx_cache)
        print(
            f"[CHECK] Decoder {position + 1}/{len(token_ids)}: "
            f"原始/改写={original_error:.8g}, "
            f"改写/ONNX={onnx_error:.8g}, Cache={cache_error:.8g}")
        if not np.allclose(original_logits, step_logits.numpy(),
                           atol=args.atol, rtol=args.rtol):
            raise ValueError(f"Decoder-Step 第 {position} 步与原始模型不一致.")
        if not np.allclose(step_logits.numpy(), onnx_logits,
                           atol=args.atol, rtol=args.rtol):
            raise ValueError(f"Decoder-Step 第 {position} 步 ONNX 不一致.")
        cache = onnx_cache
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
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    return parser.parse_args()


if __name__ == "__main__":
    run_verification(parse_args())
