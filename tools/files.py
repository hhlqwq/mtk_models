"""文件读取、完整性校验和部署所需公共模块的复制工具."""

import hashlib
import json
from pathlib import Path
import shutil


def sha256(path, *, progress=False):
    """分块计算文件哈希,可每处理 512 MiB 显示进度."""
    path = Path(path)
    digest = hashlib.sha256()
    processed = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            processed += len(block)
            if progress and processed % (512 * 1024 * 1024) == 0:
                print(f"[哈希] {path.name}: {processed // 1024**2} MiB", flush=True)
    return digest.hexdigest()


def require_file(path):
    """要求输入为存在且非空的普通文件,返回 Path."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"缺少文件或文件为空: {path}.")
    return path


def single_file(folder, pattern):
    """要求目录中存在唯一且非空的匹配文件."""
    matches = list(Path(folder).glob(pattern))
    if (len(matches) != 1 or not matches[0].is_file()
            or matches[0].stat().st_size == 0):
        raise ValueError(f"期望唯一非空文件: {folder}/{pattern}")
    return matches[0]


def read_json(path):
    """读取 UTF-8 JSON,由调用处校验各自的数据契约."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def copy_public_tools(destination, modules):
    """复制实际部署需要的公共模块,不复制源码目录或建立链接."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent
    for name in ("__init__", *modules):
        if not name.replace("_", "").isalnum():
            raise ValueError(f"公共模块名无效: {name}")
        shutil.copyfile(source / f"{name}.py", destination / f"{name}.py")
