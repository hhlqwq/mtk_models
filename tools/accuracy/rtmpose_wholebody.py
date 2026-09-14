"""执行 RTMPose PyTorch/ONNX 全量推理并生成三后端精度对比."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Callable

import cv2
import numpy as np
import tqdm


KEYPOINT_COUNT = 133
SIMCC_SPLIT_RATIO = 2.0


def load_manifest(path: Path) -> list[dict]:
    """读取板端共用的人体检测框清单.

    Args:
        path: TSV 检测框清单.

    Returns:
        按清单顺序排列的检测框.
    """
    records = []
    seen_ids = set()
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        expected_fields = {
            "image_id", "detection_id", "bbox_score", "x", "y", "w", "h"
        }
        if set(reader.fieldnames or []) != expected_fields:
            raise ValueError(f"检测框清单字段错误: {reader.fieldnames}")
        for line_number, row in enumerate(reader, start=2):
            record = {
                "image_id": int(row["image_id"]),
                "detection_id": int(row["detection_id"]),
                "bbox_score": float(row["bbox_score"]),
                "bbox": tuple(float(row[name]) for name in ("x", "y", "w", "h")),
            }
            if record["detection_id"] in seen_ids:
                raise ValueError(
                    f"第 {line_number} 行 detection_id 重复: "
                    f"{record['detection_id']}")
            if record["bbox"][2] <= 0 or record["bbox"][3] <= 0:
                raise ValueError(f"第 {line_number} 行人体框无效: {record['bbox']}")
            seen_ids.add(record["detection_id"])
            records.append(record)
    if not records:
        raise ValueError(f"检测框清单为空: {path}")
    return records


def load_completed(path: Path) -> tuple[set[int], dict[str, float]]:
    """读取可续跑 JSONL 并拒绝中间损坏或重复记录.

    Args:
        path: 后端预测 JSONL.

    Returns:
        已完成 detection_id 集合和耗时累计值.
    """
    completed = set()
    timing = {"preprocess_ms": 0.0, "inference_ms": 0.0,
              "postprocess_ms": 0.0}
    if not path.exists():
        return completed, timing
    with path.open("rb+") as input_file:
        valid_end = 0
        line_number = 0
        while True:
            line = input_file.readline()
            if not line:
                break
            line_number += 1
            try:
                item = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                if input_file.read(1):
                    raise ValueError(f"预测文件第 {line_number} 行损坏: {path}")
                input_file.truncate(valid_end)
                print(f"[RESUME] 已截断末尾未完成记录: {path}")
                break
            detection_id = int(item["detection_id"])
            if detection_id in completed:
                raise ValueError(f"预测文件存在重复 detection_id: {detection_id}")
            completed.add(detection_id)
            for name in timing:
                timing[name] += float(item.get("timing_ms", {}).get(name, 0.0))
            valid_end = input_file.tell()
            if line_number % 10000 == 0:
                print(f"[RESUME] 已核对 {line_number} 条历史预测.")
    return completed, timing


def resolve_mmpose_config(relative_path: str) -> Path:
    """从 MMPose 安装目录解析锁定的官方配置.

    Args:
        relative_path: MMPose 仓库内相对配置路径.

    Returns:
        可读取的配置文件路径.
    """
    import mmpose

    package_root = Path(mmpose.__file__).resolve().parent
    candidates = (
        package_root.parent / relative_path,
        package_root / ".mim" / relative_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"未找到 MMPose 官方配置: {relative_path}")


def create_torch_infer(
        model_dir: Path, weights: Path, config_relative: str,
        device: str) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """创建官方 PyTorch FP32 单样本推理函数.

    Args:
        model_dir: RTMPose 模型交付目录.
        weights: 官方 PyTorch 权重.
        config_relative: 官方配置相对路径.
        device: PyTorch 设备.

    Returns:
        接收 NCHW RGB FP32 输入的推理函数.
    """
    deploy_dir = model_dir / "deploy"
    sys.path.insert(0, str(deploy_dir))
    from export_onnx import (EXPECTED_MMPOSE_VERSION, EXPECTED_WEIGHTS_SHA256,
                             RTMPoseExportWrapper, prepare_mmcv_lite_import,
                             sha256_file)
    prepare_mmcv_lite_import()
    import mmpose
    import torch
    from mmpose.apis import init_model

    if mmpose.__version__ != EXPECTED_MMPOSE_VERSION:
        raise RuntimeError(
            "MMPose 版本不匹配: "
            f"expected={EXPECTED_MMPOSE_VERSION}, actual={mmpose.__version__}")
    if sha256_file(weights) != EXPECTED_WEIGHTS_SHA256:
        raise ValueError(f"RTMPose 官方权重 SHA-256 不匹配: {weights}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA 不可用,禁止静默退回 CPU.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config_path = resolve_mmpose_config(config_relative)
    model = init_model(str(config_path), str(weights), device="cpu")
    wrapper = RTMPoseExportWrapper(model).eval().to(device)

    def infer(input_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """执行一次 PyTorch FP32 推理."""
        tensor = torch.from_numpy(input_tensor).to(device)
        with torch.inference_mode():
            pred_x, pred_y = wrapper(tensor)
        return pred_x.cpu().numpy(), pred_y.cpu().numpy()

    return infer


def create_onnx_infer(
        model: Path, provider: str) -> Callable[
            [np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """创建固定 provider 的 ONNX Runtime 单样本推理函数.

    Args:
        model: FP32 ONNX 文件.
        provider: cuda 或 cpu.

    Returns:
        接收 NCHW RGB FP32 输入的推理函数.
    """
    import onnxruntime

    provider_name = {
        "cuda": "CUDAExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }[provider]
    available = onnxruntime.get_available_providers()
    if provider_name not in available:
        raise RuntimeError(
            f"ONNX Runtime provider 不可用: {provider_name}, available={available}")
    session = onnxruntime.InferenceSession(
        str(model), providers=[provider_name])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    if output_names != ["pred_x", "pred_y"]:
        raise ValueError(f"ONNX 输出名称不匹配: {output_names}")

    def infer(input_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """执行一次 ONNX Runtime FP32 推理."""
        pred_x, pred_y = session.run(output_names, {input_name: input_tensor})
        return pred_x, pred_y

    return infer


def decode_prediction(pred_x: np.ndarray, pred_y: np.ndarray,
                      metadata: dict) -> tuple[list[float], float]:
    """按板端相同的 SimCC 规则解码并映射回原图.

    Args:
        pred_x: X 轴 SimCC 输出.
        pred_y: Y 轴 SimCC 输出.
        metadata: 仿射裁剪元数据.

    Returns:
        展平的 133 点结果和裁剪框面积.
    """
    if pred_x.shape != (1, KEYPOINT_COUNT, 384):
        raise ValueError(f"pred_x 形状错误: {pred_x.shape}")
    if pred_y.shape != (1, KEYPOINT_COUNT, 512):
        raise ValueError(f"pred_y 形状错误: {pred_y.shape}")
    x_indices = pred_x[0].argmax(axis=1)
    y_indices = pred_y[0].argmax(axis=1)
    x_scores = pred_x[0, np.arange(KEYPOINT_COUNT), x_indices]
    y_scores = pred_y[0, np.arange(KEYPOINT_COUNT), y_indices]
    scores = np.minimum(x_scores, y_scores)
    center = np.asarray(metadata["center"], dtype=np.float32)
    scale = np.asarray(metadata["scale"], dtype=np.float32)
    source_x = (x_indices / SIMCC_SPLIT_RATIO / 192.0 * scale[0] +
                center[0] - scale[0] * 0.5)
    source_y = (y_indices / SIMCC_SPLIT_RATIO / 256.0 * scale[1] +
                center[1] - scale[1] * 0.5)
    source_x = np.where(scores > 0.0, source_x, -1.0)
    source_y = np.where(scores > 0.0, source_y, -1.0)
    keypoints = np.stack((source_x, source_y, scores), axis=1)
    return keypoints.astype(np.float32).reshape(-1).tolist(), float(np.prod(scale))


def run_inference(args: argparse.Namespace) -> None:
    """执行一个 FP32 后端的全量或断点续跑推理.

    Args:
        args: 命令行参数.
    """
    model_deploy = args.model_dir / "deploy"
    sys.path.insert(0, str(model_deploy))
    from rtmpose_utils import preprocess_image

    records = load_manifest(args.manifest)
    selected = records[args.start:args.start + args.total]
    if len(selected) != args.total:
        raise ValueError(
            f"请求范围超出清单: start={args.start}, total={args.total}, "
            f"manifest={len(records)}")
    selected_ids = {item["detection_id"] for item in selected}
    completed, timing = load_completed(args.predictions)
    unexpected = completed - selected_ids
    if unexpected:
        raise ValueError(
            f"已有结果包含本次范围外 detection_id: {min(unexpected)}")
    if args.backend == "pytorch":
        infer = create_torch_infer(
            args.model_dir, args.weights, args.config_relative, args.device)
    else:
        infer = create_onnx_infer(args.onnx, args.onnx_provider)

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    current_image_id = None
    current_image = None
    pending_flush = 0
    pending = [
        record for record in selected
        if record["detection_id"] not in completed
    ]
    with args.predictions.open("ab") as output_file:
        progress = tqdm.tqdm(
            pending, desc=f"RTMPose {args.backend}", unit="det",
            initial=len(completed), total=len(selected), dynamic_ncols=True)
        for record in progress:
            if current_image_id != record["image_id"]:
                image_path = args.images / f"{record['image_id']:012d}.jpg"
                current_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if current_image is None:
                    raise FileNotFoundError(f"无法读取 COCO 图片: {image_path}")
                current_image_id = record["image_id"]
            preprocess_start = time.perf_counter()
            input_tensor, metadata = preprocess_image(
                current_image, record["bbox"])
            inference_start = time.perf_counter()
            pred_x, pred_y = infer(input_tensor)
            postprocess_start = time.perf_counter()
            keypoints, area = decode_prediction(pred_x, pred_y, metadata)
            finished = time.perf_counter()
            current_timing = {
                "preprocess_ms": (inference_start - preprocess_start) * 1000.0,
                "inference_ms": (postprocess_start - inference_start) * 1000.0,
                "postprocess_ms": (finished - postprocess_start) * 1000.0,
            }
            result = {
                "detection_id": record["detection_id"],
                "image_id": record["image_id"],
                "bbox_score": record["bbox_score"],
                "area": area,
                "keypoints": keypoints,
                "timing_ms": current_timing,
            }
            output_file.write(
                (json.dumps(result, ensure_ascii=False, separators=(",", ":")) +
                 "\n").encode("utf-8"))
            completed.add(record["detection_id"])
            for name, value in current_timing.items():
                timing[name] += value
            pending_flush += 1
            if pending_flush >= args.flush_interval:
                output_file.flush()
                pending_flush = 0
        output_file.flush()
    if len(completed) != len(selected):
        raise RuntimeError(
            f"后端结果数量不完整: actual={len(completed)}, expected={len(selected)}")
    args.processed_ids.write_text(
        "".join(f"{value}\n" for value in sorted(completed)), encoding="utf-8")
    summary = {
        "backend": args.backend,
        "start": args.start,
        "total": args.total,
        "completed": len(completed),
        "timing_mean_ms": {
            name: value / len(completed) for name, value in timing.items()
        },
    }
    args.timing.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {args.backend} 推理完成: {args.predictions}")


def metric_row(name: str, path: Path) -> dict:
    """读取一个后端的 WholeBody 指标行.

    Args:
        name: 后端名称.
        path: 指标 JSON.

    Returns:
        精简指标字典.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    wholebody = payload["metrics"]["wholebody"]
    return {
        "backend": name,
        "AP": float(wholebody["AP"]),
        "AP_50": float(wholebody["AP_50"]),
        "AP_75": float(wholebody["AP_75"]),
        "AR": float(wholebody["AR"]),
        "metrics_path": str(path),
    }


def write_comparison(args: argparse.Namespace) -> None:
    """汇总 PyTorch、ONNX 和 NPU WholeBody 指标.

    Args:
        args: 命令行参数.
    """
    rows = [
        metric_row("pytorch_fp32", args.pytorch_metrics),
        metric_row("onnx_fp32", args.onnx_metrics),
        metric_row("mtk_npu_int8", args.npu_metrics),
    ]
    baseline = rows[0]
    for row in rows:
        row["delta_ap_vs_pytorch"] = row["AP"] - baseline["AP"]
        row["delta_ar_vs_pytorch"] = row["AR"] - baseline["AR"]
    payload = {"protocol": "COCO-WholeBody, same detections and postprocess",
               "backends": rows}
    args.comparison_json.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# RTMPose 三后端同协议精度对比",
        "",
        "| 后端 | WholeBody AP | AP50 | AP75 | AR | AP 相对 PyTorch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['backend']} | {row['AP']:.4f} | {row['AP_50']:.4f} | "
            f"{row['AP_75']:.4f} | {row['AR']:.4f} | "
            f"{row['delta_ap_vs_pytorch']:+.4f} |")
    args.comparison_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] 三后端精度对比: {args.comparison_json}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数.

    Returns:
        已解析参数.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--backend", choices=("pytorch", "onnx"),
                              required=True)
    infer_parser.add_argument("--model-dir", type=Path, required=True)
    infer_parser.add_argument("--weights", type=Path, required=True)
    infer_parser.add_argument("--onnx", type=Path, required=True)
    infer_parser.add_argument("--config-relative", required=True)
    infer_parser.add_argument("--manifest", type=Path, required=True)
    infer_parser.add_argument("--images", type=Path, required=True)
    infer_parser.add_argument("--predictions", type=Path, required=True)
    infer_parser.add_argument("--processed-ids", type=Path, required=True)
    infer_parser.add_argument("--timing", type=Path, required=True)
    infer_parser.add_argument("--start", type=int, default=0)
    infer_parser.add_argument("--total", type=int, default=104125)
    infer_parser.add_argument("--device", default="cuda:0")
    infer_parser.add_argument("--onnx-provider", choices=("cuda", "cpu"),
                              default="cuda")
    infer_parser.add_argument("--flush-interval", type=int, default=100)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--pytorch-metrics", type=Path, required=True)
    compare_parser.add_argument("--onnx-metrics", type=Path, required=True)
    compare_parser.add_argument("--npu-metrics", type=Path, required=True)
    compare_parser.add_argument("--comparison-json", type=Path, required=True)
    compare_parser.add_argument("--comparison-md", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "infer":
        if args.start < 0 or args.total <= 0 or args.flush_interval <= 0:
            parser.error("start 必须非负,total 和 flush-interval 必须为正整数.")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.stage == "infer":
        run_inference(arguments)
    else:
        write_comparison(arguments)
