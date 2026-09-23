"""将板端 C++ 原生 INT8 输出整理为可比较的 FP32 张量归档."""

import argparse
import json
from pathlib import Path

import numpy as np

from fastsam_utils import OUTPUT_NAMES, OUTPUT_SHAPES, sha256_file


def decode_native(path, detail):
    """按量化参数与行宽对齐契约恢复一个 NCHW 张量."""
    raw = np.fromfile(path, dtype=np.int8)
    batch, channels, height, width = map(int, detail["shape"])
    padded_width = (width + 15) // 16 * 16
    if raw.size == batch * channels * height * width:
        values = raw.reshape(batch, channels, height, width)
    elif raw.size == batch * channels * height * padded_width:
        values = raw.reshape(batch, channels, height, padded_width)[..., :width]
    else:
        raise ValueError(f"原生输出长度不匹配: {path}: {raw.size}")
    return (values.astype(np.float32) - int(detail["zero_point"])) * float(
        detail["scale"])


def main():
    """检查全部十个输出后保存同输入的浮点张量及哈希清单."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    details = {item["semantic"]: item for item in metadata["outputs"]}
    if set(details) != set(OUTPUT_NAMES):
        raise ValueError("量化元数据没有完整的十个输出.")
    with np.load(args.reference, allow_pickle=False) as reference:
        arrays = {"images": reference["images"].copy()}
    input_detail = metadata["input"]
    expected_input = np.clip(
        np.rint(arrays["images"] / float(input_detail["scale"]))
        + int(input_detail["zero_point"]), -128, 127).astype(np.int8)
    board_input_path = args.board_output / "input_int8.bin"
    board_input = np.fromfile(board_input_path, dtype=np.int8)
    if not np.array_equal(board_input, expected_input.ravel()):
        raise ValueError("板端 C++ 输入与 PyTorch 基线量化后的字节不一致.")
    digests = {}
    for name, shape in zip(OUTPUT_NAMES, OUTPUT_SHAPES):
        detail = details[name]
        if list(detail["shape"]) != shape:
            raise ValueError(f"量化元数据输出形状错误: {name}")
        path = args.board_output / f"{name}.bin"
        arrays[name] = decode_native(path, detail)
        digests[name] = sha256_file(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    args.output.with_suffix(".json").write_text(
        json.dumps({"source": "cpp_neuron_runtime_hw",
                    "board_output": str(args.board_output),
                    "raw_sha256": digests,
                    "input_sha256": sha256_file(board_input_path),
                    "input_matches_reference": True,
                    "metadata_sha256": sha256_file(args.metadata),
                    "archive_sha256": sha256_file(args.output)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[COLLECT] 已恢复 {len(digests)} 个 NPU 原始输出.")


if __name__ == "__main__":
    main()
