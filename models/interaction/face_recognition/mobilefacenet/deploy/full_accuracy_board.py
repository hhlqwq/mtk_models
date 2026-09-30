"""在 Genio 720 上完成 LFW 全量十折人脸验证。"""

import argparse
import csv
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from face_utils import load_aligned_face


EXPECTED_PAIRS = 6000


def parse_args() -> argparse.Namespace:
    """解析固定数据集、模型与运行目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_pairs(dataset_root: Path) -> list[dict[str, str]]:
    """核对完整十折 6000 对与全部已对齐图片。"""
    alignment_path = dataset_root / "alignment_summary.json"
    if not alignment_path.is_file():
        raise FileNotFoundError(f"缺少 LFW 对齐报告: {alignment_path}。")
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    if (alignment.get("status") != "complete" or
            alignment.get("aligned_images") != 7701):
        raise ValueError("LFW 对齐图未覆盖全部 7,701 张验证图。")
    with (dataset_root / "pairs.csv").open(
            newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_PAIRS:
        raise ValueError(f"LFW 验证对数量不符: {len(rows)}。")
    folds = Counter(int(row["fold_id"]) for row in rows)
    labels = Counter(int(row["is_same"]) for row in rows)
    if (folds != {fold: 600 for fold in range(1, 11)} or
            labels != {0: 3000, 1: 3000}):
        raise ValueError("LFW 十折或正负样本数量不符。")
    images_dir = dataset_root / "images"
    for row in rows:
        for field in ("image_a", "image_b"):
            name = row[field]
            if Path(name).name != name or not (images_dir / name).is_file():
                raise ValueError(f"LFW 图片缺失或路径非法: {name}。")
    return rows


def make_input(image: Path, metadata: dict, destination: Path) -> None:
    """按上游 RGB 归一化协议缩放并量化原始 LFW 图。"""
    input_meta = metadata["input"]
    if (input_meta["shape"] != [1, 3, 112, 112] or
            input_meta["dtype"] != "int8"):
        raise ValueError("输入元数据与已验证模型不符。")
    tensor = load_aligned_face(image)
    scale = float(input_meta["scale"])
    if scale <= 0:
        raise ValueError("输入量化 scale 必须大于 0。")
    quantized = np.clip(
        np.round(tensor / scale) + int(input_meta["zero_point"]),
        -128, 127).astype(np.int8)
    quantized.tofile(destination)


def load_feature(path: Path, metadata: dict) -> np.ndarray:
    """反量化并单位化板端 128 维人脸特征。"""
    output_meta = metadata["output"]
    if (output_meta["shape"] != [1, 128] or
            output_meta["dtype"] != "int8"):
        raise ValueError("输出元数据与已验证模型不符。")
    values = np.fromfile(path, dtype=np.int8)
    if values.size != 128:
        raise ValueError(f"板端人脸特征长度异常: {path}。")
    feature = ((values.astype(np.float32) - output_meta["zero_point"]) *
               output_meta["scale"])
    norm = float(np.linalg.norm(feature))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"板端人脸特征无效: {path}。")
    return feature / norm


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """仅用其他九折训练分数选择最高准确率的余弦阈值。"""
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    positive_total = int(sorted_labels.sum())
    best_correct = positive_total
    best_value = float(sorted_scores[0]) - 1e-6
    positive_below = negative_below = 0
    for index, score in enumerate(sorted_scores):
        positive_below += int(sorted_labels[index] == 1)
        negative_below += int(sorted_labels[index] == 0)
        if index + 1 < len(sorted_scores) and score == sorted_scores[index + 1]:
            continue
        correct = negative_below + positive_total - positive_below
        threshold = (float(score + sorted_scores[index + 1]) / 2.0
                     if index + 1 < len(sorted_scores) else float(score) + 1e-6)
        if correct > best_correct:
            best_correct = correct
            best_value = threshold
    return best_value


def evaluate_pairs(rows: list[dict[str, str]],
                   features: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    """计算预定义十折验证准确率与逐对判定。"""
    scores = np.array([
        float(np.dot(features[row["image_a"]], features[row["image_b"]]))
        for row in rows], dtype=np.float64)
    labels = np.array([int(row["is_same"]) for row in rows], dtype=np.int8)
    folds = np.array([int(row["fold_id"]) for row in rows], dtype=np.int8)
    details = [dict(row, cosine_similarity=float(score))
               for row, score in zip(rows, scores)]
    fold_reports = []
    for fold in range(1, 11):
        train = folds != fold
        test = folds == fold
        threshold = best_threshold(scores[train], labels[train])
        predictions = scores[test] >= threshold
        correct = int(np.count_nonzero(predictions == labels[test]))
        fold_reports.append({"fold": fold, "pairs": int(test.sum()),
                             "correct": correct, "accuracy": correct / 600,
                             "threshold": threshold})
        for detail, prediction in zip(
                (item for item in details if int(item["fold_id"]) == fold),
                predictions):
            detail["predicted_same"] = int(prediction)
    correct_total = sum(item["correct"] for item in fold_reports)
    summary = {
        "metric_protocol": "lfw_10_fold_train_9_test_1_cosine_threshold",
        "pairs": EXPECTED_PAIRS,
        "positive_pairs": int(labels.sum()),
        "negative_pairs": int(len(labels) - labels.sum()),
        "correct_pairs": correct_total,
        "verification_accuracy": correct_total / EXPECTED_PAIRS,
        "fold_mean_accuracy": float(np.mean(
            [item["accuracy"] for item in fold_reports])),
        "fold_std_accuracy": float(np.std(
            [item["accuracy"] for item in fold_reports])),
        "folds": fold_reports,
    }
    return summary, details


def evaluate(args: argparse.Namespace) -> None:
    """逐张执行硬件推理并计算全部验证对,支持安全续跑。"""
    rows = load_pairs(args.dataset_root)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata.get("input_protocol") != "RGB_ImageNet_normalize_aligned_112x112":
        raise ValueError("量化元数据未使用修正后的 RGB 输入协议。")
    source_manifest = args.dataset_root / "source_manifest.json"
    if not source_manifest.is_file() or not args.model.is_file():
        raise FileNotFoundError("缺少模型或固定数据集来源清单。")
    if args.resume:
        if not args.run_dir.is_dir():
            raise FileNotFoundError(args.run_dir)
    elif args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    for name in ("inputs", "outputs", "logs", "checkpoints", "report"):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)
    names = sorted({row[field] for row in rows
                    for field in ("image_a", "image_b")})
    features = {}
    records = {}
    wall_times = []
    for index, name in enumerate(names, start=1):
        image = args.dataset_root / "images" / name
        stem = Path(name).stem
        input_path = args.run_dir / "inputs" / f"{stem}.bin"
        output_path = args.run_dir / "outputs" / f"{stem}.bin"
        checkpoint = args.run_dir / "checkpoints" / f"{stem}.json"
        if checkpoint.exists():
            if not args.resume:
                raise FileExistsError(checkpoint)
            record = json.loads(checkpoint.read_text(encoding="utf-8"))
            if record.get("image") != name or not output_path.is_file():
                raise ValueError(f"续跑检查点不匹配: {checkpoint}。")
        else:
            if input_path.exists() or output_path.exists():
                raise ValueError(f"存在未完成的逐图文件,请先检查: {stem}。")
            make_input(image, metadata, input_path)
            started = time.perf_counter()
            with (args.run_dir / "logs" / f"{stem}.log").open("w") as log:
                subprocess.run(
                    ["/usr/sbin/neuronrt", "-m", "hw", "-a", str(args.model),
                     "-i", str(input_path), "-o", str(output_path)],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            cli_wall_ms = (time.perf_counter() - started) * 1000.0
            feature = load_feature(output_path, metadata)
            record = {
                "image": name,
                "feature_norm": float(np.linalg.norm(feature)),
                "cli_wall_ms": cli_wall_ms,
            }
            temporary = checkpoint.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
            temporary.replace(checkpoint)
        features[name] = load_feature(output_path, metadata)
        records[name] = record
        wall_times.append(float(record["cli_wall_ms"]))
        if index % 100 == 0 or index == len(names):
            print(f"[PROGRESS] MobileFaceNet {index}/{len(names)} 张。",
                  flush=True)
    verification, details = evaluate_pairs(rows, features)
    report = {
        "status": "complete",
        "model": "mobilefacenet",
        "run_id": args.run_id,
        "dataset": "LFW_upstream_MTCNN_5_point_aligned",
        "input_protocol": "RGB_ImageNet_normalize_aligned_112x112",
        "distinct_images": len(names),
        **verification,
        "cli_wall_mean_ms": float(np.mean(wall_times)),
        "cli_wall_p95_ms": float(np.percentile(wall_times, 95)),
        "timing_scope": "per-image neuronrt CLI including process startup and model load",
    }
    report_dir = args.run_dir / "report"
    (report_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    with (report_dir / "pair_results.jsonl").open("w", encoding="utf-8") as out:
        for detail in details:
            out.write(json.dumps(detail, ensure_ascii=False) + "\n")
    print(f"[OK] LFW 十折全量板端验证: {report_dir}。", flush=True)


if __name__ == "__main__":
    evaluate(parse_args())
