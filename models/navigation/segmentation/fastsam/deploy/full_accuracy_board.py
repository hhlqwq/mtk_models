"""在板端评测 FastSAM-s 的 COCO val2017 类别无关实例分割 AP."""

import argparse
import contextlib
import io
import json
import os
import statistics
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args() -> argparse.Namespace:
    """解析板端全量分割精度参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("encode-one", "evaluate"),
                        required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def make_class_agnostic_ground_truth(annotation_path: Path) -> COCO:
    """将 COCO 80 类 GT 统一映射到一个实例类别,保持原始掩码与图片."""
    source = COCO(str(annotation_path))
    dataset = dict(source.dataset)
    dataset["categories"] = [{"id": 1, "name": "object", "supercategory": "object"}]
    dataset["annotations"] = [
        {**item, "category_id": 1}
        for item in source.dataset["annotations"]
    ]
    ground_truth = COCO()
    ground_truth.dataset = dataset
    ground_truth.createIndex()
    return ground_truth


def save_checkpoint(path: Path, record: dict) -> None:
    """原子保存单图结果,避免中断后读取到不完整的 JSON."""
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def encode_one(args: argparse.Namespace) -> None:
    """将单张 C++ 推理输出编码为 COCO RLE 指标输入."""
    if args.image is None:
        raise ValueError("encode-one 需要 --image.")
    image_id = int(args.image.stem)
    image_dir = args.work_dir / "raw" / args.image.stem
    checkpoint_dir = args.work_dir / "predictions_by_image"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{args.image.stem}.json"
    if checkpoint.exists():
        raise FileExistsError(f"检查点已存在: {checkpoint}.")
    result = json.loads((image_dir / "results.json").read_text(
        encoding="utf-8"))
    if result.get("backend") != "cpp_neuron_runtime_hw":
        raise ValueError(f"不是硬件推理结果: {args.image.name}.")
    image_predictions = []
    for detection in result["detections"]:
        mask_path = image_dir / detection["mask"]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"缺少预测掩码: {mask_path}.")
        encoded = mask_utils.encode(
            np.asfortranarray((mask > 0).astype(np.uint8)))
        encoded["counts"] = encoded["counts"].decode("ascii")
        image_predictions.append({
            "image_id": image_id,
            "category_id": 1,
            "segmentation": encoded,
            "score": float(detection["score"]),
        })
    save_checkpoint(checkpoint, {
        "image_id": image_id,
        "npu_ms": float(result["npu_ms"]),
        "end_to_end_ms": float(result["end_to_end_ms"]),
        "predictions": image_predictions,
    })


def evaluate(args: argparse.Namespace) -> None:
    """核对完整 C++ 预测并计算 COCO segm AP."""
    images = sorted(args.images.glob("*.jpg"))
    if len(images) != 5000:
        raise ValueError(f"COCO val2017 需要 5000 张图片,实际 {len(images)}.")
    ground_truth = make_class_agnostic_ground_truth(args.annotations)
    expected = set(ground_truth.getImgIds())
    if {int(item.stem) for item in images} != expected:
        raise ValueError("COCO 图片与实例分割标注 ID 不一致.")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report_dir = args.work_dir / "report"
    report_dir.mkdir(exist_ok=True)
    checkpoint_dir = args.work_dir / "predictions_by_image"
    prediction_path = args.work_dir / "coco_segm_predictions.json"
    completed_path = args.work_dir / "processed_ids.txt"
    npu_times = []
    end_to_end_times = []
    prediction_count = 0

    if not checkpoint_dir.is_dir():
        raise ValueError("缺少 C++ 全量推理检查点目录.")
    with (prediction_path.open("w", encoding="utf-8") as predictions,
          completed_path.open("w", encoding="utf-8") as completed):
        predictions.write("[")
        for image in images:
            record = json.loads((checkpoint_dir / f"{image.stem}.json").read_text(
                encoding="utf-8"))
            if record["image_id"] != int(image.stem):
                raise ValueError(f"检查点图片 ID 不一致: {image.stem}")
            npu_times.append(record["npu_ms"])
            end_to_end_times.append(record["end_to_end_ms"])
            completed.write(f"{record['image_id']}\n")
            for prediction in record["predictions"]:
                if prediction_count:
                    predictions.write(",")
                predictions.write(json.dumps(prediction, ensure_ascii=False))
                prediction_count += 1
        predictions.write("]\n")

    detection = ground_truth.loadRes(str(prediction_path))
    evaluator = COCOeval(ground_truth, detection, "segm")
    evaluator.params.imgIds = sorted(expected)
    evaluator.params.catIds = [1]
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    (report_dir / "coco_summary.log").write_text(log.getvalue(), encoding="utf-8")
    summary = {
        "status": "complete",
        "model": "fastsam",
        "run_id": args.run_id,
        "dataset": "coco_val2017_instances",
        "metric_protocol": "class_agnostic_segm_ap_all_gt_categories_merged",
        "images": 5000,
        "prediction_instances": prediction_count,
        "AP_50_95": float(evaluator.stats[0]),
        "AP_50": float(evaluator.stats[1]),
        "AP_75": float(evaluator.stats[2]),
        "npu_mean_ms": statistics.fmean(npu_times),
        "npu_p95_ms": float(np.percentile(npu_times, 95)),
        "end_to_end_mean_ms": statistics.fmean(end_to_end_times),
        "timing_scope": "C++ per-image model reload; NPU time reported separately",
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[OK] FastSAM 全量类别无关分割 AP: {report_dir}.")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "encode-one":
        encode_one(arguments)
    else:
        evaluate(arguments)
