"""在 MTK 容器中生成正式精度评测 Mel、板端清单和 FP32 基线."""

import argparse
import hashlib
import json
import time
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
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--board-mel-root", required=True)
    parser.add_argument("--reference-device", choices=("none", "cpu", "cuda"),
                        default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-generated-tokens", type=int, default=196)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict]:
    """读取 JSONL 数据清单."""
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def load_pcm_wav(path: Path) -> np.ndarray:
    """读取 16 kHz 单声道 16-bit PCM WAV."""
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels() != 1 or stream.getframerate() != 16000 or
                stream.getsampwidth() != 2):
            raise ValueError(f"WAV 格式不符合要求: {path}")
        samples = np.frombuffer(
            stream.readframes(stream.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def token_config(model: whisper.Whisper, language: str,
                 max_tokens: int) -> str:
    """生成与 OpenAI Greedy Search 一致的板端 Token 规则."""
    tokenizer = get_tokenizer(
        multilingual=model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task="transcribe")
    suppressed = set(tokenizer.non_speech_tokens)
    suppressed.update((tokenizer.transcribe, tokenizer.translate,
                       tokenizer.sot, tokenizer.sot_prev, tokenizer.sot_lm))
    if tokenizer.no_speech is not None:
        suppressed.add(tokenizer.no_speech)
    suppress_first = set(tokenizer.encode(" "))
    suppress_first.add(tokenizer.eot)
    join = lambda values: ",".join(str(value) for value in values)
    return "\n".join((
        f"initial={join(tokenizer.sot_sequence_including_notimestamps)}",
        f"suppress={join(sorted(suppressed))}",
        f"suppress_first={join(sorted(suppress_first))}",
        f"eot={tokenizer.eot}",
        f"max_generated_tokens={max_tokens}",
    )) + "\n"


def prepare_mels(records: list[dict], output_dir: Path,
                 board_mel_root: str) -> None:
    """生成 FP16 Mel 和板端 TSV 清单."""
    mel_dir = output_dir / "mels"
    mel_dir.mkdir(parents=True, exist_ok=True)
    board_manifest = output_dir / "board_manifest.tsv"
    preprocess_path = output_dir / "preprocess_metrics.jsonl"
    with (board_manifest.open("w", encoding="utf-8", newline="\n") as output,
          preprocess_path.open("w", encoding="utf-8", newline="\n") as
          metrics_output):
        for index, record in enumerate(records, 1):
            decode_start = time.perf_counter()
            audio = load_pcm_wav(Path(record["audio_path"]))
            decode_end = time.perf_counter()
            mel_start = time.perf_counter()
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio, n_mels=80)
            mel_end = time.perf_counter()
            if tuple(mel.shape) != (80, 3000):
                raise ValueError(f"Mel Shape 非预期: {record['sample_id']}")
            mel_path = mel_dir / f"{record['sample_id']}.bin"
            cache_hit = mel_path.is_file() and mel_path.stat().st_size == 480000
            write_start = time.perf_counter()
            if not cache_hit:
                mel.numpy().astype(np.float16).tofile(mel_path)
            write_end = time.perf_counter()
            metrics_output.write(json.dumps({
                "sample_id": record["sample_id"],
                "duration_seconds": record["duration_seconds"],
                "audio_decode_ms": (decode_end - decode_start) * 1000.0,
                "log_mel_ms": (mel_end - mel_start) * 1000.0,
                "mel_write_ms": (write_end - write_start) * 1000.0,
                "cache_hit": cache_hit,
            }, ensure_ascii=False) + "\n")
            board_path = (
                f"{board_mel_root.rstrip('/')}/{record['sample_id']}.bin")
            output.write(
                f"{record['sample_id']}\t{board_path}\t"
                f"{record['duration_seconds']:.9f}\n")
            if index % 100 == 0 or index == len(records):
                print(f"[PROGRESS] Mel {index}/{len(records)}")


def run_reference(model: whisper.Whisper, records: list[dict], mel_dir: Path,
                  language: str, batch_size: int,
                  output_path: Path) -> None:
    """批量运行 OpenAI 模型并保存 FP32/FP16 框架基线."""
    options = DecodingOptions(
        language=language,
        task="transcribe",
        without_timestamps=True,
        fp16=next(model.parameters()).device.type == "cuda",
        temperature=0.0)
    completed = set()
    if output_path.is_file():
        completed = {json.loads(line)["sample_id"] for line in
                     output_path.read_text(encoding="utf-8").splitlines()
                     if line.strip()}
    with output_path.open("a", encoding="utf-8", newline="\n") as output:
        pending = [record for record in records
                   if record["sample_id"] not in completed]
        for begin in range(0, len(pending), batch_size):
            batch = pending[begin:begin + batch_size]
            batch_mel = torch.stack([
                torch.from_numpy(np.fromfile(
                    mel_dir / f"{record['sample_id']}.bin",
                    dtype=np.float16).reshape(80, 3000)).float()
                for record in batch
            ]).to(next(model.parameters()).device)
            with torch.no_grad():
                results = whisper.decode(model, batch_mel, options)
            for record, result in zip(batch, results):
                output.write(json.dumps({
                    "sample_id": record["sample_id"],
                    "text": result.text.strip(),
                    "tokens": [int(token) for token in result.tokens],
                }, ensure_ascii=False) + "\n")
                output.flush()
            print(f"[PROGRESS] FP32/FP16 基线 "
                  f"{min(begin + batch_size, len(pending))}/{len(pending)}")


def main() -> None:
    """生成 Mel、解码配置、板端清单和可选框架基线."""
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size 必须大于 0.")
    records = read_manifest(args.source_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu" if args.reference_device == "none" else args.reference_device
    print(f"[1/4] 加载 Whisper-Tiny: device={device}.")
    model = whisper.load_model(str(args.weights), device=device)
    print("[2/4] 生成 FP16 Mel 和板端清单.")
    prepare_mels(records, args.output_dir, args.board_mel_root)
    (args.output_dir / "decode_config.txt").write_text(
        token_config(model, args.language, args.max_generated_tokens),
        encoding="utf-8", newline="\n")
    if args.reference_device != "none":
        print("[3/4] 运行 OpenAI 框架基线.")
        run_reference(model, records, args.output_dir / "mels",
                      args.language, args.batch_size,
                      args.output_dir / "reference_predictions.jsonl")
    else:
        print("[3/4] 跳过 OpenAI 框架基线.")
    print("[4/4] 写入输入清单哈希.")
    hashes = {
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "board_manifest_sha256": sha256_file(
            args.output_dir / "board_manifest.tsv"),
        "decode_config_sha256": sha256_file(
            args.output_dir / "decode_config.txt"),
        "sample_count": len(records),
        "language": args.language,
        "reference_device": args.reference_device,
    }
    (args.output_dir / "input_manifest.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[OK] 正式评测输入: {args.output_dir}")


if __name__ == "__main__":
    main()
