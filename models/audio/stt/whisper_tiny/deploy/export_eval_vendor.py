"""导出板端 WER 所需的固定版 OpenAI Whisper 文本组件。"""

import argparse
import hashlib
import json
import shutil
from importlib import metadata
from pathlib import Path

import whisper


SOURCE_FILES = (
    "tokenizer.py",
    "assets/multilingual.tiktoken",
    "normalizers/__init__.py",
    "normalizers/basic.py",
    "normalizers/english.py",
    "normalizers/english.json",
)


def sha256_file(path: Path) -> str:
    """逐块计算导出文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_vendor(output_dir: Path) -> None:
    """复制原版文本组件并记录版本、来源和文件哈希。"""
    version = metadata.version("openai-whisper")
    if version != "20250625":
        raise ValueError(f"OpenAI Whisper 版本不符: {version}。")
    source_root = Path(whisper.__file__).resolve().parent
    package_root = output_dir / "whisper"
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    files = {}
    for relative in SOURCE_FILES:
        source = source_root / relative
        destination = package_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files[relative] = sha256_file(destination)
        print(f"[EXPORT] {relative}", flush=True)
    manifest = {
        "source": "https://github.com/openai/whisper",
        "package": "openai-whisper",
        "version": version,
        "purpose": "LibriSpeech test-clean Token 解码与英文 WER 规范化",
        "files_sha256": files,
    }
    (output_dir / "vendor_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def main() -> None:
    """解析导出目录并生成可审计的板端依赖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_vendor(args.output_dir)


if __name__ == "__main__":
    main()
