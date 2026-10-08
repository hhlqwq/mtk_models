"""导出板端 WER 所需的固定版 OpenAI Whisper 文本组件。"""

import argparse
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


def export_vendor(output_dir: Path) -> None:
    """复制固定版本的原版文本组件。"""
    version = metadata.version("openai-whisper")
    if version != "20250625":
        raise ValueError(f"OpenAI Whisper 版本不符: {version}。")
    source_root = Path(whisper.__file__).resolve().parent
    package_root = output_dir / "whisper"
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    for relative in SOURCE_FILES:
        source = source_root / relative
        destination = package_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"[EXPORT] {relative}", flush=True)


def main() -> None:
    """解析导出目录并生成可审计的板端依赖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_vendor(args.output_dir)


if __name__ == "__main__":
    main()
