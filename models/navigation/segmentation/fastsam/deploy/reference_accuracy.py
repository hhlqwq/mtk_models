"""使用 FP32 ONNX 完成 COCO val2017 类别无关分割参考精度。"""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval

from fastsam_utils import postprocess, preprocess
from full_accuracy_board import make_class_agnostic_ground_truth


def sha256_file(path: Path) -> str:
    """逐块计算模型与标注文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """核对全量图片，运行 FP32 模型并按板端规则计算 segm AP。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    images = sorted(args.images.glob("*.jpg"))
    ground_truth = make_class_agnostic_ground_truth(args.annotations)
    expected = set(ground_truth.getImgIds())
    if len(images) != 5000 or {int(image.stem) for image in images} != expected:
        raise ValueError("COCO val2017 图片或标注不完整。")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True)
    session = ort.InferenceSession(
        str(args.model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("FP32 ONNX CUDA EP 不可用。")
    input_name = session.get_inputs()[0].name
    predictions = []
    for index, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"图片无法解码: {image_path}。")
        tensor, geometry = preprocess(image)
        outputs = session.run(None, {input_name: tensor})
        _, scores, masks = postprocess(outputs, geometry, 0.4, 0.9, 100)
        for score, mask in zip(scores, masks):
            encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            encoded["counts"] = encoded["counts"].decode("ascii")
            predictions.append({"image_id": int(image_path.stem), "category_id": 1,
                                "segmentation": encoded, "score": float(score)})
        if index % 100 == 0 or index == 5000:
            print(f"[PROGRESS] FastSAM FP32 {index}/5000 张。", flush=True)
    prediction_path = args.work_dir / "coco_segm_predictions.json"
    prediction_path.write_text(json.dumps(predictions, ensure_ascii=False),
                               encoding="utf-8")
    result = ground_truth.loadRes(str(prediction_path))
    evaluator = COCOeval(ground_truth, result, "segm")
    evaluator.params.imgIds = sorted(expected)
    evaluator.params.catIds = [1]
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    (args.output_dir / "cocoeval.log").write_text(log.getvalue(), encoding="utf-8")
    report = {"status": "complete", "backend": "onnxruntime_fp32_cuda_preferred",
              "dataset": "coco_val2017_class_agnostic_segm",
              "images": len(images), "predictions": len(predictions),
              "ap50_95": float(evaluator.stats[0]),
              "ap50": float(evaluator.stats[1]),
              "model_sha256": sha256_file(args.model),
              "annotations_sha256": sha256_file(args.annotations),
              "predictions_sha256": sha256_file(prediction_path)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[RESULT] FastSAM FP32 AP50:95 {report['ap50_95']:.6f}。", flush=True)


if __name__ == "__main__":
    main()
