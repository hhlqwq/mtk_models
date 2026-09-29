"""以板端相同解码协议评测 YOLO-World XL FP32 ONNX 全量 bbox AP。"""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import cv2
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yoloworld_utils import COCO_CLASSES, decode_outputs, preprocess_image


def sha256_file(path: Path) -> str:
    """逐块核对模型、标注和全量预测。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """核对 COCO 5,000 张，运行浮点模型并计算 bbox AP。"""
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
    annotation = COCO(str(args.annotations))
    expected = set(annotation.getImgIds())
    if len(images) != 5000 or {int(item.stem) for item in images} != expected:
        raise ValueError("COCO val2017 图片或标注不完整。")
    categories = {item["name"]: item["id"]
                  for item in annotation.dataset["categories"]}
    if set(COCO_CLASSES) != set(categories):
        raise ValueError("固化的 COCO 80 类与标注不一致。")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True)
    session = ort.InferenceSession(
        str(args.model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("FP32 ONNX CUDA EP 不可用。")
    input_name = session.get_inputs()[0].name
    predictions = []
    for index, path in enumerate(images, 1):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"图片无法解码: {path}。")
        tensor, transform = preprocess_image(image)
        outputs = session.run(None, {input_name: tensor})
        detections = decode_outputs(outputs, transform, score_threshold=0.001,
                                    iou_threshold=0.65, max_detections=300)
        for item in detections:
            left, top, right, bottom = item["bbox_xyxy"]
            predictions.append({"image_id": int(path.stem),
                                "category_id": categories[item["class_name"]],
                                "bbox": [left, top, right - left, bottom - top],
                                "score": item["score"]})
        if index % 100 == 0 or index == 5000:
            print(f"[PROGRESS] YOLO-World FP32 {index}/5000 张。", flush=True)
    prediction_path = args.work_dir / "coco_predictions.json"
    prediction_path.write_text(json.dumps(predictions, ensure_ascii=False),
                               encoding="utf-8")
    result = annotation.loadRes(str(prediction_path))
    evaluator = COCOeval(annotation, result, "bbox")
    evaluator.params.imgIds = sorted(expected)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    (args.output_dir / "cocoeval.log").write_text(log.getvalue(), encoding="utf-8")
    report = {"status": "complete", "backend": "onnxruntime_fp32_cuda_preferred",
              "dataset": "coco_val2017_bbox_80_classes", "images": 5000,
              "predictions": len(predictions),
              "ap50_95": float(evaluator.stats[0]),
              "ap50": float(evaluator.stats[1]),
              "model_sha256": sha256_file(args.model),
              "annotations_sha256": sha256_file(args.annotations),
              "predictions_sha256": sha256_file(prediction_path)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[RESULT] YOLO-World FP32 AP50:95 {report['ap50_95']:.6f}。",
          flush=True)


if __name__ == "__main__":
    main()
