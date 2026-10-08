"""在 89 导出 C++ 板端预处理和解码所需的静态 Whisper 资源."""

import argparse
from pathlib import Path

import numpy as np
import torch
import whisper
from whisper.tokenizer import get_tokenizer


def parse_args() -> argparse.Namespace:
    """解析静态资源输出目录."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def export_assets(output_dir: Path) -> None:
    """导出官方 80 Mel 滤波器和英文 Greedy 解码规则,不运行推理."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filters = whisper.audio.mel_filters(torch.device("cpu"), 80)
    if tuple(filters.shape) != (80, 201):
        raise ValueError(f"Whisper Mel 滤波器形状异常: {filters.shape}.")
    np.asarray(filters, dtype="<f4").tofile(output_dir / "mel_filters_f32.bin")

    tokenizer = get_tokenizer(multilingual=True, language="en",
                              task="transcribe")
    suppressed = set(tokenizer.non_speech_tokens)
    suppressed.update((tokenizer.transcribe, tokenizer.translate,
                       tokenizer.sot, tokenizer.sot_prev, tokenizer.sot_lm))
    if tokenizer.no_speech is not None:
        suppressed.add(tokenizer.no_speech)
    suppress_first = set(tokenizer.encode(" "))
    suppress_first.add(tokenizer.eot)
    join = lambda values: ",".join(map(str, values))
    config = "\n".join((
        f"initial={join(tokenizer.sot_sequence_including_notimestamps)}",
        f"suppress={join(sorted(suppressed))}",
        f"suppress_first={join(sorted(suppress_first))}",
        f"eot={tokenizer.eot}",
        "max_generated_tokens=196",
    )) + "\n"
    (output_dir / "decode_config.txt").write_text(
        config, encoding="utf-8")
    print(f"[OK] C++ Whisper 静态资源: {output_dir}.")


if __name__ == "__main__":
    export_assets(parse_args().output_dir)
