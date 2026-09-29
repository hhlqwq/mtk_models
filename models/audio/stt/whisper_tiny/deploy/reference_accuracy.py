"""在 LibriSpeech test-clean 全量数据上评测 OpenAI Whisper-Tiny 参考 WER。"""

import argparse
import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import torch
import whisper

from evaluate_accuracy import metric_summary


EXPECTED_SAMPLES = 2620


def sha256_file(path: Path) -> str:
    """逐块计算权重或记录文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sources(dataset_root: Path) -> dict[str, dict]:
    """从 test-clean 官方转录文件核对全部音频与参考文本。"""
    sources = {}
    for transcript in sorted(dataset_root.rglob("*.trans.txt")):
        for line in transcript.read_text(encoding="utf-8").splitlines():
            sample_id, reference_text = line.split(" ", 1)
            audio = transcript.parent / f"{sample_id}.flac"
            if sample_id in sources or not audio.is_file():
                raise ValueError(f"重复样例或缺少音频: {sample_id}。")
            sources[sample_id] = {"sample_id": sample_id,
                                  "reference_text": reference_text,
                                  "audio": audio}
    if len(sources) != EXPECTED_SAMPLES:
        raise ValueError(f"test-clean 样例不完整: {len(sources)}。")
    return sources


def load_cached_audio(path: Path) -> np.ndarray:
    """按 OpenAI Whisper 的 PCM16 输入比例读取 16 kHz 单声道缓存。"""
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels() != 1 or stream.getframerate() != 16000 or
                stream.getsampwidth() != 2):
            raise ValueError(f"音频缓存格式异常: {path}。")
        samples = np.frombuffer(stream.readframes(stream.getnframes()),
                                dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def main() -> None:
    """固定 30 秒 Mel 与英文 Greedy 解码，保存全量预测和 WER。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--audio-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    sources = load_sources(args.dataset_root)
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    predictions = {}
    if args.resume and prediction_path.exists():
        for line in prediction_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item["sample_id"] in predictions:
                raise ValueError(f"重复预测: {item['sample_id']}。")
            predictions[item["sample_id"]] = item
    model = whisper.load_model(str(args.weights), device="cuda")
    model.eval()
    options = whisper.DecodingOptions(
        task="transcribe", language="en", without_timestamps=True,
        temperature=0, sample_len=196, fp16=True)
    with prediction_path.open("a", encoding="utf-8") as output:
        for index, (sample_id, source) in enumerate(sorted(sources.items()), 1):
            if sample_id in predictions:
                continue
            audio = load_cached_audio(args.audio_cache / f"{sample_id}.wav")
            if not np.isfinite(audio).all():
                raise ValueError(f"音频无效: {sample_id}。")
            mel = whisper.log_mel_spectrogram(
                whisper.pad_or_trim(audio), n_mels=model.dims.n_mels).to("cuda")
            with torch.inference_mode():
                decoded = whisper.decode(model, mel, options)
            record = {"sample_id": sample_id, "text": decoded.text.strip(),
                      "tokens": list(decoded.tokens)}
            predictions[sample_id] = record
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if index % 25 == 0 or index == len(sources):
                print(f"[PROGRESS] Whisper FP32 {index}/{len(sources)} 条。",
                      flush=True)
    if set(predictions) != set(sources):
        raise ValueError("test-clean 参考预测覆盖不完整。")
    accuracy, details = metric_summary("librispeech", sources, predictions)
    report = {"status": "complete", "backend": "openai_whisper_cuda_fp16_compute",
              "dataset": "librispeech_test_clean_30s_window",
              "expected_samples": EXPECTED_SAMPLES,
              "successful_samples": len(predictions),
              "weights_sha256": sha256_file(args.weights),
              "predictions_sha256": sha256_file(prediction_path),
              "accuracy": accuracy}
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "worst_samples.jsonl").open("w", encoding="utf-8") as output:
        for item in sorted(details, key=lambda value: value["error_rate"],
                           reverse=True)[:100]:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[RESULT] test-clean 参考 WER {accuracy['value']:.6f}。", flush=True)


if __name__ == "__main__":
    main()
