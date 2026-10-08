"""核查 MobileFaceNet 板端输出是否构成有效冒烟证据。"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_embedding(path: Path, output: dict) -> np.ndarray:
    """读取并反量化单次板端输出。"""
    dtype = np.dtype(output["dtype"])
    raw = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(output["shape"]))
    if raw.size != expected:
        raise ValueError(f"输出长度异常: {path}, {raw.size} != {expected}")
    result = (raw.astype(np.float32) - output["zero_point"]) * output["scale"]
    if not np.isfinite(result).all() or np.linalg.norm(result) <= 0:
        raise ValueError(f"输出非有限数或全零: {path}")
    return result


def verify(metadata_path: Path, output_dir: Path, result_path: Path) -> None:
    """验证三次真实推理并记录相似度与重复输入一致性。"""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output = metadata["output"]
    vectors = [load_embedding(output_dir / f"{sample['stem']}.bin", output)
               for sample in metadata["samples"]]
    if len(vectors) != 3:
        raise ValueError("必须包含两个不同输入和一次重复输入")
    repeated_equal = bool(np.array_equal(vectors[0], vectors[2]))
    if not repeated_equal:
        raise ValueError("同一输入的两次板端输出不一致")
    cosine = float(np.dot(vectors[0], vectors[1]) /
                   (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])))
    result = {
        "status": "smoke_passed",
        "embedding_elements": int(vectors[0].size),
        "repeated_input_equal": repeated_equal,
        "different_input_cosine": cosine,
        "scope": "仅验证特征提取推理，不代表身份识别准确率",
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    """解析板端证据路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    verify(args.metadata, args.output_dir, args.result)


if __name__ == "__main__":
    main()
