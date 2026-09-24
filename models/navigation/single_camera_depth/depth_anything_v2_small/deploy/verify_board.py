"""核对板端深度输出、重复性与同输入 PyTorch 参考。"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from depth_utils import INPUT_ROW_STRIDE, INPUT_SIZE, sha256_file


def load_depth(path: Path, scale: float, zero_point: int) -> tuple[np.ndarray, int]:
    """解析 INT8 原始输出并记录可能的行步长。"""
    raw = np.fromfile(path, dtype=np.int8)
    expected = INPUT_SIZE * INPUT_SIZE
    if raw.size < expected or raw.size % INPUT_SIZE:
        raise ValueError(f"输出尺寸异常: {path}，{raw.size} 字节")
    row_stride = raw.size // INPUT_SIZE
    values = raw.reshape(INPUT_SIZE, row_stride)[:, :INPUT_SIZE]
    depth = (values.astype(np.float32) - zero_point) * scale
    if not np.isfinite(depth).all() or depth.std() <= 0:
        raise ValueError(f"输出不是有限的非恒定深度图: {path}")
    return depth, row_stride


def save_visualization(depth: np.ndarray, path: Path) -> None:
    """保存按单图范围归一化的彩色预览。"""
    low, high = np.percentile(depth, [2, 98])
    if high <= low:
        raise ValueError("深度图动态范围不足。")
    normalized = np.clip((depth - low) * 255 / (high - low),
                         0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO))


def verify_run(run_dir: Path, metadata_path: Path, upstream: Path,
               weights: Path) -> dict:
    """检查三次硬件输出并与原始 PyTorch 模型逐图比较。"""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output_meta = metadata["output"]
    input_meta = metadata["input"]
    if output_meta["shape"] != [1, INPUT_SIZE, INPUT_SIZE]:
        raise ValueError(f"输出形状异常: {output_meta['shape']}")
    manifest = json.loads((run_dir / "inputs/manifest.json").read_text(
        encoding="utf-8"))
    raw_paths = [run_dir / "output" / f"image_{index}.bin"
                 for index in (1, 2, 3)]
    if raw_paths[0].read_bytes() != raw_paths[2].read_bytes():
        raise ValueError("同一输入的两次板端原始输出不一致。")
    if raw_paths[0].read_bytes() == raw_paths[1].read_bytes():
        raise ValueError("不同图片得到完全相同的板端输出。")

    sys.path.insert(0, str(upstream))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    model.eval()
    samples = []
    for index in (1, 2):
        image_meta = manifest[index - 1]
        input_path = run_dir / "inputs" / f"image_{index}.bin"
        quantized = np.fromfile(input_path, dtype=np.int8)
        if quantized.size != 3 * INPUT_SIZE * INPUT_ROW_STRIDE:
            raise ValueError(f"输入长度异常: {input_path}")
        tensor = ((quantized.astype(np.float32) - input_meta["zero_point"])
                  * input_meta["scale"]).reshape(
                      1, 3, INPUT_SIZE, INPUT_ROW_STRIDE)[..., :INPUT_SIZE]
        with torch.no_grad():
            reference = model(torch.from_numpy(tensor))[0].numpy()
        board, row_stride = load_depth(
            raw_paths[index - 1], output_meta["scale"],
            output_meta["zero_point"])
        correlation = float(np.corrcoef(reference.ravel(), board.ravel())[0, 1])
        if not np.isfinite(correlation) or correlation < 0.9:
            raise ValueError(
                f"image_{index} 的 PyTorch/板端深度相关系数过低: {correlation}")
        save_visualization(board, run_dir / "output" / f"image_{index}_depth.png")
        samples.append({
            "image": image_meta["image"],
            "image_sha256": image_meta["image_sha256"],
            "input_sha256": sha256_file(input_path),
            "output_sha256": sha256_file(raw_paths[index - 1]),
            "output_row_stride": row_stride,
            "board_min": float(board.min()),
            "board_max": float(board.max()),
            "board_std": float(board.std()),
            "pytorch_min": float(reference.min()),
            "pytorch_max": float(reference.max()),
            "pytorch_std": float(reference.std()),
            "pytorch_board_pearson": correlation,
        })
        print(f"[VERIFY] image_{index}: Pearson={correlation:.6f}", flush=True)
    report = {
        "status": "smoke_passed",
        "input_size": INPUT_SIZE,
        "repeat_raw_output_identical": True,
        "different_images_output_distinct": True,
        "weights_sha256": sha256_file(weights),
        "onnx_sha256": metadata["onnx_sha256"],
        "tflite_sha256": metadata["tflite_sha256"],
        "samples": samples,
    }
    (run_dir / "output/report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """解析运行路径并执行冒烟核验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    verify_run(args.run_dir, args.metadata, args.upstream, args.weights)


if __name__ == "__main__":
    main()
