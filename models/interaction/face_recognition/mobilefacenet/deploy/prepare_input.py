"""为板端推理生成量化人脸输入和张量元数据。"""

import argparse
import json
from pathlib import Path

import mtk_converter
import numpy as np

from face_utils import load_aligned_face


def tensor_metadata(detail: dict) -> dict:
    """提取张量形状、类型和逐张量量化参数。"""
    scales = detail["quantization"]["scales"]
    zero_points = detail["quantization"]["zero_points"]
    if len(scales) != 1 or len(zero_points) != 1:
        raise ValueError(f"不支持该张量量化配置: {detail['name']}")
    return {
        "shape": [int(value) for value in detail["shape"]],
        "dtype": np.dtype(detail["dtype"]).name,
        "scale": float(scales[0]),
        "zero_point": int(zero_points[0]),
    }


def prepare_inputs(tflite: Path, image_dir: Path, output_dir: Path) -> None:
    """生成两张不同人脸和一张重复人脸的冒烟输入。"""
    parser = mtk_converter.TFLiteParser(str(tflite))
    input_detail = parser.get_input_tensor_details()[0]
    outputs = parser.get_output_tensor_details()
    input_meta = tensor_metadata(input_detail)
    if input_meta["shape"] != [1, 3, 112, 112]:
        raise ValueError(f"输入形状异常: {input_meta['shape']}")
    if input_meta["dtype"] != "int8" or len(outputs) != 1:
        raise ValueError("冒烟程序要求单输入 INT8、单输出图")
    output_meta = tensor_metadata(outputs[0])
    images = sorted(image_dir.glob("*_aligned.jpg"))[:2]
    if len(images) != 2:
        raise ValueError("至少需要两张对齐人脸")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"input": input_meta, "output": output_meta, "samples": []}
    for index, path in enumerate((images[0], images[1], images[0]), start=1):
        tensor = load_aligned_face(path)
        scale = input_meta["scale"]
        if scale <= 0:
            raise ValueError("输入量化 scale 必须大于 0")
        quantized = np.clip(
            np.round(tensor / scale) + input_meta["zero_point"],
            -128, 127).astype(np.int8)
        stem = f"face_{index}"
        quantized.tofile(output_dir / f"{stem}.bin")
        metadata["samples"].append({"stem": stem, "image": str(path)})
        print(f"[PREPARE] {index}/3 {path.name}")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] 板端输入: {output_dir}")


def main() -> None:
    """解析输入准备参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tflite", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_inputs(args.tflite, args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()
