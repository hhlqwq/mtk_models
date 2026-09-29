"""在 Genio 720 上评测 DA-2K 全量相对深度点对准确率。"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from depth_utils import INPUT_ROW_STRIDE, INPUT_SIZE, preprocess


EXPECTED_IMAGES = 1033
EXPECTED_PAIRS = 2068


def sha256_file(path: Path) -> str:
    """逐块计算文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    """解析固定数据集、模型与本次运行目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_protocol(dataset_root: Path) -> dict[str, list[dict]]:
    """核对官方压缩包中全部图片与点对标注。"""
    path = dataset_root / "annotations.json"
    annotations = json.loads(path.read_text(encoding="utf-8"))
    if len(annotations) != EXPECTED_IMAGES:
        raise ValueError(f"DA-2K 图片数不符: {len(annotations)}。")
    pair_count = sum(len(pairs) for pairs in annotations.values())
    if pair_count != EXPECTED_PAIRS:
        raise ValueError(f"DA-2K 点对数不符: {pair_count}。")
    for relative, pairs in annotations.items():
        if not relative.startswith("images/") or ".." in Path(relative).parts:
            raise ValueError(f"非法图片路径: {relative}。")
        if not (dataset_root / relative).is_file():
            raise FileNotFoundError(dataset_root / relative)
        if not pairs:
            raise ValueError(f"缺少深度点对: {relative}。")
        if any(pair.get("closer_point") != "point1" for pair in pairs):
            raise ValueError(f"深度标注方向异常: {relative}。")
    return annotations


def make_input(image: Path, metadata: dict, destination: Path) -> None:
    """按已验证的 518x518 量化协议生成 MDLA 行补齐输入。"""
    input_meta = metadata["input"]
    if (input_meta["shape"] != [1, 3, INPUT_SIZE, INPUT_SIZE] or
            input_meta["dtype"] != "int8"):
        raise ValueError("输入元数据与已验证模型不符。")
    tensor = preprocess(image)
    scale = float(input_meta["scale"])
    if scale <= 0:
        raise ValueError("输入量化 scale 必须大于 0。")
    zero_point = int(input_meta["zero_point"])
    quantized = np.clip(np.round(tensor / scale) + zero_point,
                        -128, 127).astype(np.int8)
    padded = np.pad(quantized,
                    ((0, 0), (0, 0), (0, 0),
                     (0, INPUT_ROW_STRIDE - INPUT_SIZE)),
                    mode="constant", constant_values=zero_point)
    padded.tofile(destination)


def load_output(path: Path, metadata: dict) -> np.ndarray:
    """反量化原始输出并去除可能存在的行补齐。"""
    output_meta = metadata["output"]
    if (output_meta["shape"] != [1, INPUT_SIZE, INPUT_SIZE] or
            output_meta["dtype"] != "int8"):
        raise ValueError("输出元数据与已验证模型不符。")
    raw = np.fromfile(path, dtype=np.int8)
    if raw.size < INPUT_SIZE * INPUT_SIZE or raw.size % INPUT_SIZE:
        raise ValueError(f"板端输出长度异常: {path}。")
    depth = raw.reshape(INPUT_SIZE, -1)[:, :INPUT_SIZE]
    values = ((depth.astype(np.float32) - output_meta["zero_point"]) *
              output_meta["scale"])
    if not np.isfinite(values).all() or values.std() <= 0:
        raise ValueError(f"板端输出无效或恒定: {path}。")
    return values


def score_pairs(image: Path, depth: np.ndarray,
                pairs: list[dict]) -> tuple[int, int]:
    """按原图坐标比较点对,数值较大的一点判为更近。"""
    original = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"无法读取图片: {image}。")
    height, width = original.shape[:2]
    restored = cv2.resize(depth, (width, height),
                          interpolation=cv2.INTER_LINEAR)
    correct = 0
    for pair in pairs:
        points = []
        for key in ("point1", "point2"):
            y, x = pair[key]
            if not (0 <= y < height and 0 <= x < width):
                raise ValueError(f"标注坐标越界: {image}, {key}。")
            points.append(float(restored[int(y), int(x)]))
        correct += points[0] > points[1]
    return int(correct), len(pairs)


def evaluate(args: argparse.Namespace) -> None:
    """逐图完成硬件推理,支持同一运行编号的安全续跑。"""
    annotations = load_protocol(args.dataset_root)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if args.resume:
        if not args.run_dir.is_dir():
            raise FileNotFoundError(args.run_dir)
    elif args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    for name in ("inputs", "outputs", "logs", "checkpoints", "report"):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)

    totals = {}
    for index, relative in enumerate(sorted(annotations), start=1):
        image = args.dataset_root / relative
        stem = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
        input_path = args.run_dir / "inputs" / f"{stem}.bin"
        output_path = args.run_dir / "outputs" / f"{stem}.bin"
        checkpoint = args.run_dir / "checkpoints" / f"{stem}.json"
        if checkpoint.exists():
            if not args.resume:
                raise FileExistsError(checkpoint)
            record = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (record.get("image") != relative or
                    record.get("output_sha256") != sha256_file(output_path)):
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
            depth = load_output(output_path, metadata)
            correct, total = score_pairs(image, depth, annotations[relative])
            record = {
                "image": relative,
                "image_sha256": sha256_file(image),
                "input_sha256": sha256_file(input_path),
                "output_sha256": sha256_file(output_path),
                "correct_pairs": correct,
                "total_pairs": total,
                "cli_wall_ms": cli_wall_ms,
            }
            temporary = checkpoint.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
            temporary.replace(checkpoint)
        totals[relative] = record
        if index % 25 == 0 or index == EXPECTED_IMAGES:
            print(f"[PROGRESS] DA-2K {index}/{EXPECTED_IMAGES} 张。", flush=True)

    correct_pairs = sum(item["correct_pairs"] for item in totals.values())
    total_pairs = sum(item["total_pairs"] for item in totals.values())
    if total_pairs != EXPECTED_PAIRS:
        raise ValueError(f"实际评测点对数不符: {total_pairs}。")
    scenes = {}
    for relative, record in totals.items():
        scene = Path(relative).parts[1]
        current = scenes.setdefault(scene, {"images": 0, "correct_pairs": 0,
                                            "total_pairs": 0})
        current["images"] += 1
        current["correct_pairs"] += record["correct_pairs"]
        current["total_pairs"] += record["total_pairs"]
    for item in scenes.values():
        item["pairwise_accuracy"] = (item["correct_pairs"] /
                                      item["total_pairs"])
    wall_times = np.array([item["cli_wall_ms"] for item in totals.values()])
    report = {
        "status": "complete",
        "model": "depth_anything_v2_small",
        "run_id": args.run_id,
        "dataset": "DA-2K",
        "metric_protocol": "fixed_518_square_bilinear_restore_point1_closer",
        "images": len(totals),
        "correct_pairs": correct_pairs,
        "total_pairs": total_pairs,
        "pairwise_accuracy": correct_pairs / total_pairs,
        "scenes": scenes,
        "cli_wall_mean_ms": float(wall_times.mean()),
        "cli_wall_p95_ms": float(np.percentile(wall_times, 95)),
        "timing_scope": "per-image neuronrt CLI including process startup and model load",
        "model_sha256": sha256_file(args.model),
        "metadata_sha256": sha256_file(args.metadata),
        "annotations_sha256": sha256_file(args.dataset_root / "annotations.json"),
    }
    report_path = args.run_dir / "report" / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    image_hashes = args.run_dir / "report" / "dataset_images_sha256.txt"
    image_hashes.write_text("".join(
        f"{totals[relative]['image_sha256']}  {relative}\n"
        for relative in sorted(totals)), encoding="utf-8")
    output_hashes = args.run_dir / "report" / "raw_outputs_sha256.txt"
    output_hashes.write_text("".join(
        f"{totals[relative]['output_sha256']}  {relative}\n"
        for relative in sorted(totals)), encoding="utf-8")
    print(f"[OK] DA-2K 全量点对准确率: {report_path}。", flush=True)


if __name__ == "__main__":
    evaluate(parse_args())
