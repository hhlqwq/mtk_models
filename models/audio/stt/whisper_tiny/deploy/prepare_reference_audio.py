"""在 89 宿主机将 LibriSpeech test-clean 全量转换成 Whisper PCM 缓存。"""

import argparse
import subprocess
import wave
from pathlib import Path

EXPECTED_SAMPLES = 2620


def load_sources(dataset_root: Path) -> dict[str, Path]:
    """核对官方转录清单与对应 FLAC 文件。"""
    sources = {}
    for transcript in sorted(dataset_root.rglob("*.trans.txt")):
        for line in transcript.read_text(encoding="utf-8").splitlines():
            sample_id = line.split(" ", 1)[0]
            audio = transcript.parent / f"{sample_id}.flac"
            if sample_id in sources or not audio.is_file():
                raise ValueError(f"重复样例或音频缺失: {sample_id}。")
            sources[sample_id] = audio
    if len(sources) != EXPECTED_SAMPLES:
        raise ValueError(f"test-clean 样例数不符: {len(sources)}。")
    return sources


def main() -> None:
    """逐条使用固定 ffmpeg 参数生成 16 kHz 单声道 PCM16 WAV。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = load_sources(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (sample_id, source) in enumerate(sorted(sources.items()), 1):
        output = args.output_dir / f"{sample_id}.wav"
        if not output.exists():
            subprocess.run([
                "ffmpeg", "-nostdin", "-threads", "0", "-i", str(source),
                "-f", "wav", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                "-loglevel", "error", str(output)], check=True)
        with wave.open(str(output), "rb") as stream:
            if (stream.getnchannels() != 1 or stream.getframerate() != 16000 or
                    stream.getsampwidth() != 2 or stream.getnframes() == 0):
                raise ValueError(f"音频缓存不完整: {output}。")
        if index % 100 == 0 or index == EXPECTED_SAMPLES:
            print(f"[PROGRESS] 音频缓存 {index}/{EXPECTED_SAMPLES} 条。", flush=True)


if __name__ == "__main__":
    main()
