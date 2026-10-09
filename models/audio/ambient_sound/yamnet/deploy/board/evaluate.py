"""板端音频前处理、严格全量校验、精度汇总和实际声音示例."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from audio_utils import load_records, load_waveform, mapping, metrics, preprocess


def prepare(args):
    """在板端从真实 WAV 生成窗口缓存并记录逐音频前处理耗时."""
    records = load_records(args.dataset)
    args.work.mkdir(parents=True, exist_ok=False)
    rows = []
    with (args.work / "features.bin").open("wb") as features, \
            (args.work / "patch_counts.txt").open("w") as counts:
        for index, row in enumerate(records):
            start = time.perf_counter()
            waveform = load_waveform(args.dataset / "audio" / row["filename"])
            patches = preprocess(waveform)
            elapsed = (time.perf_counter() - start) * 1000
            features.write(patches.astype("<f4").tobytes())
            counts.write(f"{len(patches)}\n")
            rows.append({"filename": row["filename"], "patches": len(patches),
                         "audio_seconds": len(waveform) / 16000,
                         "preprocess_ms": elapsed,
                         "sha256": hashlib.sha256(
                             (args.dataset / "audio" / row["filename"]).read_bytes()).hexdigest()})
            if (index + 1) % 50 == 0:
                print(f"[AUDIO] {index + 1}/2000 音频.", flush=True)
    (args.work / "audio_manifest.json").write_text(json.dumps(rows, indent=2))
    (args.work / "frontend_peak_rss_kib.txt").write_text(
        str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print("[OK] 板端真实音频窗口缓存完成.", flush=True)


def summarize(args):
    """验证覆盖和参考协议,计算主指标、窗口性能与可读示例."""
    records = load_records(args.dataset)
    labels = args.models_dir / "yamnet_class_map.csv"
    indices = mapping(labels)
    scores = np.fromfile(args.work / "scores.bin", dtype="<f4")
    if scores.size != 2000 * 521 or not np.isfinite(scores).all():
        raise ValueError("板端分数覆盖不足或含非有限数.")
    scores = scores.reshape(2000, 521)
    audio = json.loads((args.work / "audio_manifest.json").read_text())
    if [row["filename"] for row in audio] != [row["filename"] for row in records]:
        raise ValueError("音频顺序不一致.")
    with (args.work / "timings.csv").open() as stream:
        timing_rows = list(csv.DictReader(stream))
    expected = [(clip, patch) for clip, row in enumerate(audio)
                for patch in range(row["patches"])]
    if [(int(row["clip"]), int(row["patch"])) for row in timing_rows] != expected:
        raise ValueError("窗口耗时覆盖不足、重复或顺序错误.")
    npu = np.array([float(row["npu_ms"]) for row in timing_rows])
    if not np.isfinite(npu).all() or np.any(npu <= 0):
        raise ValueError("NPU 耗时含无效数.")
    reference = json.loads((args.models_dir / "fp32_reference.json").read_text())
    annotation_hash = hashlib.sha256((args.dataset / "meta/esc50.csv").read_bytes()).hexdigest()
    if reference["annotations_sha256"] != annotation_hash or reference["mapping"] != indices:
        raise ValueError("主机与板端标注或标签映射不一致.")
    if {row["filename"]: row["sha256"] for row in audio} != reference["audio_sha256"]:
        raise ValueError("主机与板端原始音频哈希不一致.")
    actual = metrics(scores, records, indices)
    baseline = reference["onnx"]["metrics"]["held_out"]["map"]
    primary = actual["held_out"]["map"]
    with (args.work / "clip_times.csv").open() as stream:
        clip_rows = list(csv.DictReader(stream))
    if [int(row["clip"]) for row in clip_rows] != list(range(2000)):
        raise ValueError("音频推理阶段耗时覆盖不足.")
    preprocess_ms = np.array([row["preprocess_ms"] for row in audio])
    inference_ms = np.array([float(row["inference_stage_ms"]) for row in clip_rows])
    config_name = f"runtime_config_{args.precision}.csv"
    if not (args.models_dir / config_name).exists():
        if args.precision == "w8a16":
            raise FileNotFoundError("W8A16 必须使用独立的真实 INT16 IO 配置.")
        config_name = "runtime_config.csv"
    summary = {
        "status": "complete", "model": "yamnet", "run_id": args.run_id,
        "precision": args.precision,
        "dataset": "ESC-50", "samples": 2000, "patches": len(npu),
        "metric": "ESC-50 projected macro AP, folds 2-5",
        "evaluation_samples": actual["held_out"]["samples"],
        "calibration_fold": 1 if args.precision != "fp16" else None,
        "evaluation_folds": [2, 3, 4, 5],
        "mapped_classes": 47,
        "unmapped_classes": ["drinking_sipping", "can_opening", "washing_machine"],
        "board_accuracy": primary, "reference_accuracy": baseline,
        "accuracy_change_percentage_points": (primary - baseline) * 100,
        "reference_source": "本次 ONNX FP32 同协议实测,固定标签投影,窗口分数均值",
        "metrics": actual, "tensorflow_metrics": reference["tensorflow"]["metrics"],
        "onnx_metrics": reference["onnx"]["metrics"],
        "npu_mean_ms": float(npu.mean()), "npu_p95_ms": float(np.percentile(npu, 95)),
        "timing_scope": "常驻模型每个 0.48 秒步长窗口的 NeuronRuntime_inference,排除 20 次预热",
        "preprocess_mean_ms_per_clip": float(preprocess_ms.mean()),
        "inference_stage_mean_ms_per_clip": float(inference_ms.mean()),
        "processing_real_time_factor": float((preprocess_ms.sum() + inference_ms.sum()) /
                                             (sum(row["audio_seconds"] for row in audio) * 1000)),
        "processing_scope": "分阶段前处理与推理处理时间之和,含 ffmpeg 解码及特征缓存读取,排除 20 次预热;非实时麦克风端到端延迟",
        "peak_rss_mib": int((args.work / "peak_rss_kib.txt").read_text()) / 1024,
        "frontend_peak_rss_mib": int((args.work / "frontend_peak_rss_kib.txt").read_text()) / 1024,
        "memory_scope": "NPU C++ 进程和 Python 前处理进程峰值分别记录,不包含 ffmpeg 子进程",
        "annotations_sha256": annotation_hash,
        "artifacts_sha256": {
            name: hashlib.sha256((args.models_dir / name).read_bytes()).hexdigest()
            for name in (f"model_{args.precision}.dla", config_name, "yamnet_class_map.csv")},
        "runtime_program_sha256": hashlib.sha256(
            (args.models_dir.parent / "board/yamnet_eval").read_bytes()).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with labels.open() as stream:
        class_names = [row["display_name"] for row in csv.DictReader(stream)]
    examples = args.output / "examples/output"
    examples.mkdir(parents=True, exist_ok=True)
    for number, target in enumerate((0, 39, 42), start=1):
        index = next(index for index, row in enumerate(records) if int(row["target"]) == target)
        order = np.argsort(-scores[index])[:5]
        result = {"source_filename": records[index]["filename"],
                  "esc50_category": records[index]["category"],
                  "top5_yamnet": [{"class": class_names[int(category)],
                                   "score": float(scores[index, category])} for category in order]}
        (examples / f"sample_{number}_events.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[OK] {args.output / 'summary.json'}", flush=True)


def main():
    """运行板端准备或最终汇总,各阶段失败时保留现场."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "summarize"], required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--precision", choices=["fp16", "int8", "w8a16"], default="fp16")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        if not all((args.models_dir, args.output, args.run_id)):
            parser.error("汇总必须提供模型目录、输出目录和运行 ID.")
        summarize(args)


if __name__ == "__main__":
    main()
