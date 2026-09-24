"""按转换所得量化参数准备两张图片和重复输入。"""

import argparse
import json
from pathlib import Path

import numpy as np

from depth_utils import INPUT_ROW_STRIDE, INPUT_SIZE, preprocess, sha256_file


def prepare_inputs(metadata_path: Path, images: list[Path],
                   output_dir: Path) -> None:
    """生成三次推理输入并保存对应图片来源。"""
    if len(images) != 2 or images[0] == images[1]:
        raise ValueError("需要两张不同的输入图片。")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    input_meta = metadata["input"]
    if input_meta["shape"] != [1, 3, INPUT_SIZE, INPUT_SIZE]:
        raise ValueError("量化输入形状异常。")
    scale = input_meta["scale"]
    zero_point = input_meta["zero_point"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, path in enumerate([*images, images[0]], start=1):
        tensor = preprocess(path)
        quantized = np.clip(np.round(tensor / scale) + zero_point,
                            -128, 127).astype(np.int8)
        quantized = np.pad(
            quantized,
            ((0, 0), (0, 0), (0, 0), (0, INPUT_ROW_STRIDE - INPUT_SIZE)),
            mode="constant", constant_values=zero_point)
        stem = f"image_{index}"
        destination = output_dir / f"{stem}.bin"
        quantized.tofile(destination)
        manifest.append({"stem": stem, "image": str(path),
                         "image_sha256": sha256_file(path),
                         "input_sha256": sha256_file(destination),
                         "input_row_stride": INPUT_ROW_STRIDE})
        print(f"[PREPARE] {index}/3 {path.name}: {destination}", flush=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """读取量化元数据与输入图片路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image", type=Path, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_inputs(args.metadata, args.image, args.output_dir)


if __name__ == "__main__":
    main()
