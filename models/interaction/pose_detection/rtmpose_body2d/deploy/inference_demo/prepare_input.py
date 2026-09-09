"""为 RTMPose 板端推理准备 NCHW INT8 输入."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import mtk_converter
import numpy as np

DEPLOY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY_DIR))

from rtmpose_utils import load_person_samples, preprocess_image  # noqa: E402


def find_demo_samples(annotations: Path, count: int) -> list[dict]:
    """选择来自不同图片且面积较大的人体框作为稳定冒烟样本."""
    samples = []
    for sample in load_person_samples(annotations):
        x, y, width, height = sample["bbox"]
        image_width = sample["image_width"]
        image_height = sample["image_height"]
        area_ratio = sample["area"] / (image_width * image_height)
        aspect_ratio = height / width
        is_inside = (
            x >= image_width * 0.01 and y >= image_height * 0.01 and
            x + width <= image_width * 0.99 and
            y + height <= image_height * 0.99)
        if (is_inside and 0.05 <= area_ratio <= 0.50 and
                1.4 <= aspect_ratio <= 4.0 and
                width >= 64 and height >= 128):
            samples.append(sample)
    samples.sort(key=lambda item: (-item["area"], item["annotation_id"]))
    selected = []
    image_ids = set()
    for sample in samples:
        if sample["image_id"] in image_ids:
            continue
        selected.append(sample)
        image_ids.add(sample["image_id"])
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(f"独立 Demo 人体样本不足: 需要 {count}, 实际 {len(selected)}")
    return selected


def quantize_input(input_tensor: np.ndarray, detail: dict) -> np.ndarray:
    """按 TFLite 输入量化参数将 FP32 输入转换为整数张量."""
    scales = detail["quantization"]["scales"]
    zero_points = detail["quantization"]["zero_points"]
    if len(scales) != 1 or len(zero_points) != 1:
        raise ValueError(f"仅支持 per-tensor 输入量化: {detail['quantization']}")
    scale = float(scales[0])
    zero_point = int(zero_points[0])
    dtype = np.dtype(np.int8)
    limits = np.iinfo(dtype)
    return np.clip(np.round(input_tensor / scale) + zero_point,
                   limits.min, limits.max).astype(dtype)


def prepare_inputs(args: argparse.Namespace) -> None:
    """从 COCO person 框生成两份输入、元数据和原图副本."""
    parser = mtk_converter.TFLiteParser(str(args.tflite))
    input_detail = parser.get_input_tensor_details()[0]
    output_details = parser.get_output_tensor_details()
    if list(input_detail["shape"]) != [1, 3, 256, 192]:
        raise ValueError(f"TFLite 输入 shape 异常: {input_detail['shape']}")
    expected_shapes = {(1, 133, 384), (1, 133, 512)}
    actual_shapes = {
        tuple(int(value) for value in detail["shape"])
        for detail in output_details
    }
    if len(output_details) != 2 or actual_shapes != expected_shapes:
        raise ValueError(f"TFLite 输出结构异常: {output_details}")

    samples = find_demo_samples(args.annotations, args.count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input": {
            "shape": [int(value) for value in input_detail["shape"]],
            "dtype": "int8",
            "scale": float(input_detail["quantization"]["scales"][0]),
            "zero_point": int(
                input_detail["quantization"]["zero_points"][0]),
        },
        "outputs": [{
            "index": index,
            "name": detail["name"],
            "shape": [int(value) for value in detail["shape"]],
            "dtype": "int8",
            "scale": float(detail["quantization"]["scales"][0]),
            "zero_point": int(detail["quantization"]["zero_points"][0]),
        } for index, detail in enumerate(output_details)],
        "samples": [],
    }
    for index, sample in enumerate(samples):
        image_path = args.image_dir / sample["file_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取 Demo 图片: {image_path}")
        input_tensor, geometry = preprocess_image(image, sample["bbox"])
        quantized = quantize_input(input_tensor, input_detail)
        stem = f"sample_{index + 1}"
        quantized.tofile(args.output_dir / f"{stem}.bin")
        source_copy = args.output_dir / f"{stem}.jpg"
        if not cv2.imwrite(str(source_copy), image):
            raise ValueError(f"无法写入 Demo 原图: {source_copy}")
        metadata["samples"].append({
            "stem": stem,
            "source_image": str(image_path),
            "source_copy": source_copy.name,
            "annotation_id": sample["annotation_id"],
            "image_size": [sample["image_width"], sample["image_height"]],
            **geometry,
        })
        print(f"[PREPARE] {index + 1}/{len(samples)} {image_path.name}")
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] RTMPose Demo 输入: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--tflite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_inputs(parse_args())
