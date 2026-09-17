"""在 Ubuntu89 主机上构建 Whisper 正式精度评测音频清单."""

import argparse
import hashlib
import json
import shutil
import subprocess
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("librispeech", "aishell1"),
                        required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256,并显示大文件进度."""
    digest = hashlib.sha256()
    processed = 0
    next_report = 1024 * 1024 * 1024
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            processed += len(block)
            if processed >= next_report:
                print(f"[HASH] {path.name}: {processed / 2**30:.1f} GiB")
                next_report += 1024 * 1024 * 1024
    return digest.hexdigest()


def read_wav_duration(path: Path) -> float:
    """读取标准 PCM WAV 时长并校验格式."""
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels() != 1 or stream.getframerate() != 16000 or
                stream.getsampwidth() != 2):
            raise ValueError(f"WAV 格式不是 16 kHz 单声道 16-bit PCM: {path}")
        return stream.getnframes() / stream.getframerate()


def load_librispeech(root: Path) -> list[tuple[str, Path, str]]:
    """读取 LibriSpeech test-clean transcript 与 FLAC 路径."""
    samples: list[tuple[str, Path, str]] = []
    for transcript_path in sorted(root.rglob("*.trans.txt")):
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            sample_id, reference = line.split(maxsplit=1)
            audio_path = transcript_path.parent / f"{sample_id}.flac"
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            samples.append((sample_id, audio_path, reference))
    if not samples:
        raise ValueError(f"没有找到 LibriSpeech transcript: {root}")
    return samples


def load_aishell(root: Path) -> list[tuple[str, Path, str]]:
    """读取 AISHELL-1 test transcript 与 WAV 路径."""
    transcript_candidates = sorted(root.rglob("aishell_transcript*.txt"))
    if len(transcript_candidates) != 1:
        raise ValueError(
            f"AISHELL transcript 数量不是 1: {transcript_candidates}")
    references = {}
    for line in transcript_candidates[0].read_text(
            encoding="utf-8").splitlines():
        fields = line.split()
        references[fields[0]] = "".join(fields[1:])
    test_directories = [path for path in root.rglob("test") if path.is_dir()]
    wav_paths = sorted({path for directory in test_directories
                        for path in directory.rglob("*.wav")})
    samples = []
    for audio_path in wav_paths:
        sample_id = audio_path.stem
        if sample_id not in references:
            raise KeyError(f"AISHELL transcript 缺少样例: {sample_id}")
        samples.append((sample_id, audio_path, references[sample_id]))
    if not samples:
        raise ValueError(f"没有找到 AISHELL-1 test WAV: {root}")
    return samples


def stage_audio(source: Path, target: Path, dataset: str,
                ffmpeg: str) -> None:
    """复制或转换一条音频,不修改原始数据."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        read_wav_duration(target)
        return
    if dataset == "librispeech":
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(source), "-ar", "16000", "-ac", "1", "-c:a",
            "pcm_s16le", str(target)
        ], check=True)
    else:
        shutil.copy2(source, target)
    read_wav_duration(target)


def build_manifest(args: argparse.Namespace) -> None:
    """构建标准 WAV、JSONL 清单和数据来源元数据."""
    root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = (load_librispeech(root) if args.dataset == "librispeech"
               else load_aishell(root))
    print(f"[INFO] {args.dataset}: {len(samples)} 条样例.")

    manifest_path = output_dir / "source_manifest.jsonl"
    excluded_path = output_dir / "excluded_over_30s.jsonl"
    included = 0
    excluded = 0
    with (manifest_path.open("w", encoding="utf-8", newline="\n") as output,
          excluded_path.open("w", encoding="utf-8", newline="\n") as
          excluded_output):
        for index, (sample_id, source, reference) in enumerate(samples, 1):
            target = audio_dir / f"{sample_id}.wav"
            stage_audio(source, target, args.dataset, args.ffmpeg)
            duration = read_wav_duration(target)
            record = {
                "sample_id": sample_id,
                "audio_path": str(target),
                "audio_sha256": sha256_file(target),
                "duration_seconds": duration,
                "reference_text": reference,
            }
            if duration > 30.0:
                record["reason"] = "模型首版固定输入上限为 30 秒"
                excluded_output.write(
                    json.dumps(record, ensure_ascii=False) + "\n")
                excluded += 1
            else:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                included += 1
            if index % 100 == 0 or index == len(samples):
                print(f"[PROGRESS] 音频清单 {index}/{len(samples)}")

    metadata = {
        "dataset": args.dataset,
        "dataset_root": str(root),
        "source_sample_count": len(samples),
        "evaluated_sample_count": included,
        "excluded_over_30s_count": excluded,
        "source_manifest_sha256": sha256_file(manifest_path),
        "archive": str(args.archive.resolve()) if args.archive else None,
        "archive_sha256": sha256_file(args.archive.resolve())
        if args.archive else None,
        "source_page_url": (
            "https://www.openslr.org/12/"
            if args.dataset == "librispeech" else
            "https://www.openslr.org/33/"),
        "download_url": (
            "https://www.openslr.org/resources/12/test-clean.tar.gz"
            if args.dataset == "librispeech" else
            "https://openslr.magicdatatech.com/resources/33/data_aishell.tgz"),
        "license": "CC BY 4.0" if args.dataset == "librispeech"
        else "Apache License 2.0",
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[OK] 数据清单: {manifest_path}")


def main() -> None:
    """执行数据清单构建."""
    build_manifest(parse_args())


if __name__ == "__main__":
    main()
