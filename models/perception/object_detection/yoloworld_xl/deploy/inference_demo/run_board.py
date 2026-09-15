#!/usr/bin/env python3
"""在 Genio 板端通过 ONNX Runtime 执行 YOLO-World XL。"""

import argparse
import hashlib
import json
import platform
import resource
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from yoloworld_utils import decode_outputs, preprocess_image, render_detections


def parse_args() -> argparse.Namespace:
    """解析板端推理参数。"""
    parser = argparse.ArgumentParser(description="YOLO-World XL 板端推理。")
    parser.add_argument("--model", type=Path, required=True, help="ONNX 模型路径。")
    parser.add_argument("--images", type=Path, required=True, help="图片或图片目录。")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录。")
    parser.add_argument(
        "--provider",
        choices=("cpu", "neuron"),
        default="neuron",
        help="执行提供器。",
    )
    parser.add_argument("--warmup", type=int, default=3, help="预热次数。")
    parser.add_argument("--repeat", type=int, default=10, help="每张图重复推理次数。")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.65)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--profile", action="store_true", help="保存 ORT profiling。")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def collect_images(path: Path) -> list[Path]:
    """收集单张图片或目录中的图片。"""
    if path.is_file():
        return [path]
    images = sorted(
        item for item in path.iterdir() if item.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise FileNotFoundError(f"没有找到图片: {path}。")
    return images


def create_session(
    model_path: Path,
    provider: str,
    profile: bool,
    output_dir: Path,
) -> ort.InferenceSession:
    """创建 CPU 或 Neuron EP 会话。"""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile:
        options.enable_profiling = True
        options.profile_file_prefix = str(output_dir / f"ort_{provider}")
    if provider == "cpu":
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
    else:
        providers = ["NeuronExecutionProvider", "CPUExecutionProvider"]
        provider_options = [
            {
                "NEURON_FLAG_USE_FP16": "1",
                "NEURON_FLAG_MIN_GROUP_SIZE": "1",
            },
            {},
        ]
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=providers,
        provider_options=provider_options,
    )
    if session.get_providers()[0] != providers[0]:
        raise RuntimeError(
            f"请求 {providers[0]}，实际 providers={session.get_providers()}。"
        )
    return session


def percentile(values: list[float], percentage: float) -> float:
    """使用线性插值计算百分位数。"""
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentage))


def summarize_profile(profile_path: Path) -> dict[str, object]:
    """统计 ORT profiling 中各执行提供器的节点数量。"""
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    provider_counts: dict[str, int] = {}
    for event in events:
        provider = event.get("args", {}).get("provider")
        if provider:
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
    return {
        "path": str(profile_path),
        "provider_node_events": provider_counts,
    }


def main() -> None:
    """执行图片推理、后处理、可视化和性能统计。"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    images = collect_images(args.images)
    print(f"[1/4] 创建 {args.provider} 会话，模型首次建图可能耗时较长。")
    session_start = time.perf_counter()
    session = create_session(args.model, args.provider, args.profile, args.output_dir)
    session_creation_ms = (time.perf_counter() - session_start) * 1000.0

    first_image = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise ValueError(f"无法读取图片: {images[0]}。")
    warmup_input, _ = preprocess_image(first_image)
    print(f"[2/4] 预热 {args.warmup} 次。")
    for index in range(args.warmup):
        session.run(None, {"images": warmup_input})
        print(f"[WARMUP] {index + 1}/{args.warmup}")

    print(f"[3/4] 处理 {len(images)} 张图片，每张重复 {args.repeat} 次。")
    all_inference_ms: list[float] = []
    image_results = []
    for image_index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}。")
        preprocess_start = time.perf_counter()
        tensor, transform = preprocess_image(image)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        outputs = None
        image_inference_ms = []
        for _ in range(args.repeat):
            inference_start = time.perf_counter()
            outputs = session.run(None, {"images": tensor})
            elapsed_ms = (time.perf_counter() - inference_start) * 1000.0
            image_inference_ms.append(elapsed_ms)
            all_inference_ms.append(elapsed_ms)
        if outputs is None:
            raise RuntimeError("没有产生推理输出。")

        postprocess_start = time.perf_counter()
        detections = decode_outputs(
            outputs,
            transform,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            max_detections=args.max_detections,
        )
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
        rendered = render_detections(image, detections)
        output_image = args.output_dir / f"{image_path.stem}_detections.jpg"
        cv2.imwrite(str(output_image), rendered)
        image_results.append(
            {
                "image": image_path.name,
                "image_sha256": sha256_file(image_path),
                "preprocess_ms": preprocess_ms,
                "inference_ms": image_inference_ms,
                "postprocess_ms": postprocess_ms,
                "detection_count": len(detections),
                "detections": detections,
                "rendered_image": output_image.name,
            }
        )
        print(
            f"[PROGRESS] {image_index}/{len(images)}，"
            f"detections={len(detections)}，"
            f"inference_mean={statistics.fmean(image_inference_ms):.3f} ms。"
        )

    profile_summary = None
    if args.profile:
        profile_path = Path(session.end_profiling())
        profile_summary = summarize_profile(profile_path)
        if args.provider == "neuron" and not any(
            "Neuron" in name
            for name in profile_summary["provider_node_events"]
        ):
            raise RuntimeError("profiling 中没有 Neuron EP 节点执行证据。")

    timing_summary = {
        "count": len(all_inference_ms),
        "mean_ms": statistics.fmean(all_inference_ms),
        "min_ms": min(all_inference_ms),
        "max_ms": max(all_inference_ms),
        "p50_ms": percentile(all_inference_ms, 50),
        "p90_ms": percentile(all_inference_ms, 90),
        "p95_ms": percentile(all_inference_ms, 95),
    }
    report = {
        "schema_version": 1,
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "provider_requested": args.provider,
        "providers_registered": session.get_providers(),
        "onnxruntime_version": ort.__version__,
        "platform": platform.platform(),
        "session_creation_ms": session_creation_ms,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "timing": timing_summary,
        "profile": profile_summary,
        "images": image_results,
    }
    report_path = args.output_dir / "results.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("[4/4] 板端结果已保存。")
    print(json.dumps(timing_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
