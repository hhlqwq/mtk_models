"""在全量板端推理前校验 OpenAI Whisper 的轻量评测依赖。"""

import hashlib
import json
from pathlib import Path

from whisper.normalizers import EnglishTextNormalizer
from whisper.tokenizer import get_tokenizer


def sha256_file(path: Path) -> str:
    """逐块计算依赖文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_vendor(tool_dir: Path) -> None:
    """核对来源清单，并实际运行 Token 解码和英文规范化。"""
    manifest = json.loads((tool_dir / "vendor_manifest.json").read_text(
        encoding="utf-8"))
    if manifest.get("version") != "20250625":
        raise ValueError("OpenAI Whisper 评测组件版本不符。")
    for relative, expected in manifest["files_sha256"].items():
        path = tool_dir / "whisper" / relative
        if sha256_file(path) != expected:
            raise ValueError(f"评测组件 SHA-256 不符: {relative}")
    tokenizer = get_tokenizer(
        multilingual=True, language="en", task="transcribe")
    sample = "hello world"
    if tokenizer.decode(tokenizer.encode(sample)) != sample:
        raise ValueError("Token 编解码检查失败。")
    if EnglishTextNormalizer()(sample) != sample:
        raise ValueError("英文文本规范化检查失败。")
    print("[OK] OpenAI Whisper 评测组件与实际编解码检查通过。", flush=True)


if __name__ == "__main__":
    check_vendor(Path(__file__).resolve().parent)
