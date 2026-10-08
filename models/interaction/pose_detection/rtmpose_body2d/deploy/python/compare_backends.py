"""比较 RTMPose FP32 ONNX 与 MTK NPU 双输出数值和关键点位置."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime

from postprocess_keypoints import decode_simcc, load_output


def load_fp32_input(path: Path, metadata: dict) -> np.ndarray:
    """读取板端 INT8 输入并按记录的量化参数还原 FP32."""
    detail = metadata["input"]
    quantized = np.fromfile(path, dtype=np.int8)
    shape = tuple(detail["shape"])
    if quantized.size != int(np.prod(shape)):
        raise ValueError(f"输入大小异常: {path}, 实际 {quantized.size}")
    return ((quantized.reshape(shape).astype(np.float32) -
             detail["zero_point"]) * detail["scale"])


def compare_outputs(reference: np.ndarray,
                    npu: np.ndarray) -> dict[str, float]:
    """计算单个 SimCC 输出的绝对误差统计."""
    difference = np.abs(reference.astype(np.float64) -
                        npu.astype(np.float64))
    return {
        "mean_abs": float(difference.mean()),
        "max_abs": float(difference.max()),
    }


def confidence_metrics(
        distances: np.ndarray, matches: np.ndarray,
        scores: np.ndarray) -> dict[str, dict[str, float | int | None]]:
    """按 ONNX 置信度阈值统计关键点位置一致性."""
    metrics = {}
    for threshold in (0.1, 0.2):
        valid = scores >= threshold
        count = int(valid.sum())
        metrics[f"onnx_score_gte_{threshold:.1f}"] = {
            "count": count,
            "mean_distance_input_pixels": (
                float(distances[valid].mean()) if count else None),
            "max_distance_input_pixels": (
                float(distances[valid].max()) if count else None),
            "argmax_xy_match_ratio": (
                float(matches[valid].mean()) if count else None),
        }
    return metrics


def compare(args: argparse.Namespace) -> None:
    """逐样本比较 ONNX 与 NPU 输出并写入 JSON 报告."""
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    session = onnxruntime.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    output_details = metadata["outputs"]
    x_detail = next(item for item in output_details
                    if item["shape"][-1] == 384)
    y_detail = next(item for item in output_details
                    if item["shape"][-1] == 512)
    results = []
    for sample in metadata["samples"]:
        stem = sample["stem"]
        input_tensor = load_fp32_input(args.input_dir / f"{stem}.bin",
                                       metadata)
        onnx_values = session.run(output_names, {input_name: input_tensor})
        onnx_by_name = dict(zip(output_names, onnx_values))
        onnx_x = onnx_by_name["pred_x"]
        onnx_y = onnx_by_name["pred_y"]
        npu_x = load_output(
            args.output_dir / f"{stem}_{x_detail['index']}.bin", x_detail)
        npu_y = load_output(
            args.output_dir / f"{stem}_{y_detail['index']}.bin", y_detail)
        onnx_points, onnx_scores = decode_simcc(onnx_x, onnx_y)
        npu_points, npu_scores = decode_simcc(npu_x, npu_y)
        distances = np.linalg.norm(onnx_points - npu_points, axis=1)
        argmax_match = np.all(onnx_points == npu_points, axis=1)
        result = {
            "stem": stem,
            "pred_x": compare_outputs(onnx_x, npu_x),
            "pred_y": compare_outputs(onnx_y, npu_y),
            "keypoint_distance_input_pixels": {
                "mean_all_133": float(distances.mean()),
                "max_all_133": float(distances.max()),
                "mean_body_17": float(distances[:17].mean()),
                "max_body_17": float(distances[:17].max()),
            },
            "argmax_xy_match_ratio": float(argmax_match.mean()),
            "confidence_filtered": confidence_metrics(
                distances, argmax_match, onnx_scores),
            "score_mean": {
                "onnx": float(onnx_scores.mean()),
                "npu": float(npu_scores.mean()),
            },
        }
        results.append(result)
        print(
            f"[COMPARE] {stem}: body17_mean="
            f"{result['keypoint_distance_input_pixels']['mean_body_17']:.3f}"
            f" px, argmax_match={result['argmax_xy_match_ratio']:.3f}")
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps({"samples": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[OK] ONNX/NPU 对比报告: {args.result}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    compare(parse_args())
