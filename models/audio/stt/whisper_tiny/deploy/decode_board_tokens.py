"""将板端 Whisper Token ID 解码为文本并与 OpenAI 基线比较."""

import argparse
import json
from pathlib import Path

import whisper
from whisper.tokenizer import get_tokenizer


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--expected-tokens", type=Path, required=True)
    parser.add_argument("--expected-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="en")
    return parser.parse_args()


def read_tokens(path: Path) -> list[int]:
    """读取每行一个整数的 Token 文件."""
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    """解码板端 Token 并生成精确对比 JSON."""
    args = parse_args()
    checkpoint = whisper.load_model(str(args.weights), device="cpu")
    tokenizer = get_tokenizer(
        multilingual=checkpoint.is_multilingual,
        num_languages=checkpoint.num_languages,
        language=args.language,
        task="transcribe")
    actual_tokens = read_tokens(args.tokens)
    expected_tokens = read_tokens(args.expected_tokens)
    actual_text = tokenizer.decode(actual_tokens).strip()
    expected_text = args.expected_text.read_text(encoding="utf-8").strip()
    comparison = {
        "actual_text": actual_text,
        "expected_text": expected_text,
        "text_exact_match": actual_text == expected_text,
        "token_exact_match": actual_tokens == expected_tokens,
        "actual_tokens": actual_tokens,
        "expected_tokens": expected_tokens,
    }
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if not comparison["text_exact_match"]:
        raise ValueError("板端文本与 OpenAI FP32 基线不一致.")


if __name__ == "__main__":
    main()
