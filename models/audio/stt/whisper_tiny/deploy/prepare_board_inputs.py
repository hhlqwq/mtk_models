"""准备 Whisper-Tiny 板端 FP16 Mel、解码配置和 OpenAI 基线."""

import argparse
import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import torch
import whisper
from whisper.decoding import DecodingOptions
from whisper.tokenizer import get_tokenizer


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-generated-tokens", type=int, default=64)
    return parser.parse_args()


def join_tokens(tokens: list[int] | tuple[int, ...]) -> str:
    """将 Token ID 序列编码为逗号分隔文本."""
    return ",".join(str(token) for token in tokens)


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_audio_file(path: Path) -> np.ndarray:
    """读取 16 kHz 单声道 PCM WAV,其他格式交给 Whisper/ffmpeg."""
    if path.suffix.lower() != ".wav":
        return whisper.load_audio(str(path))
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels() != 1 or stream.getframerate() != 16000 or
                stream.getsampwidth() != 2):
            raise ValueError("WAV 必须为 16 kHz、单声道、16-bit PCM.")
        samples = np.frombuffer(
            stream.readframes(stream.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def prepare_inputs(args: argparse.Namespace) -> None:
    """生成板端输入、解码规则和 OpenAI FP32 参考结果."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[1/4] 加载 OpenAI Whisper-Tiny FP32 模型.")
    model = whisper.load_model(str(args.weights), device="cpu")
    tokenizer = get_tokenizer(
        multilingual=model.is_multilingual,
        num_languages=model.num_languages,
        language=args.language,
        task="transcribe")

    print("[2/4] 生成固定 30 秒 Log-Mel 并转换为 FP16.")
    audio = load_audio_file(args.audio)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
    if tuple(mel.shape) != (model.dims.n_mels, 3000):
        raise ValueError(f"Mel Shape 非预期: {tuple(mel.shape)}")
    mel.numpy().astype(np.float16).tofile(args.output_dir / "mel_fp16.bin")

    print("[3/4] 运行 OpenAI FP32 Greedy Search 基线.")
    options = DecodingOptions(
        language=args.language,
        task="transcribe",
        without_timestamps=True,
        fp16=False,
        temperature=0.0)
    with torch.no_grad():
        result = whisper.decode(model, mel, options)
    expected_tokens = [int(token) for token in result.tokens]
    (args.output_dir / "expected_tokens.txt").write_text(
        "\n".join(str(token) for token in expected_tokens) + "\n",
        encoding="utf-8")
    (args.output_dir / "expected_text.txt").write_text(
        result.text.strip() + "\n", encoding="utf-8")

    print("[4/4] 固化板端 Greedy Search Token 过滤规则.")
    suppress_tokens = set(tokenizer.non_speech_tokens)
    suppress_tokens.update((tokenizer.transcribe, tokenizer.translate,
                            tokenizer.sot, tokenizer.sot_prev,
                            tokenizer.sot_lm))
    if tokenizer.no_speech is not None:
        suppress_tokens.add(tokenizer.no_speech)
    suppress_first = set(tokenizer.encode(" "))
    suppress_first.add(tokenizer.eot)
    config_lines = [
        f"initial={join_tokens(tokenizer.sot_sequence_including_notimestamps)}",
        f"suppress={join_tokens(sorted(suppress_tokens))}",
        f"suppress_first={join_tokens(sorted(suppress_first))}",
        f"eot={tokenizer.eot}",
        f"max_generated_tokens={args.max_generated_tokens}",
    ]
    (args.output_dir / "decode_config.txt").write_text(
        "\n".join(config_lines) + "\n", encoding="utf-8")
    metadata = {
        "audio": str(args.audio),
        "audio_sha256": sha256_file(args.audio),
        "language": args.language,
        "mel_shape": [1, model.dims.n_mels, 3000],
        "mel_dtype": "float16",
        "reference_text": result.text.strip(),
        "reference_tokens": expected_tokens,
    }
    (args.output_dir / "reference.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[OK] OpenAI 基线文本: {result.text.strip()}")


def main() -> None:
    """执行板端输入准备流程."""
    prepare_inputs(parse_args())


if __name__ == "__main__":
    main()
