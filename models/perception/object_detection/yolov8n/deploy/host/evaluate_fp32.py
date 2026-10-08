"""使用统一协议评测原始 PyTorch 或自导出 ONNX 的 COCO 精度."""

import argparse
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from export_model import load_model, raw_forward

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from yolov8_utils import COCO_IDS, OUTPUT_NAMES, postprocess, preprocess


def main():
    """逐图推理并校验全部标注覆盖,保存同协议精度及预测."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.onnx) == bool(args.weights):
        raise ValueError("必须且只能选择 ONNX 或 PyTorch 权重.")
    torch.set_num_threads(2)
    annotation = COCO(str(args.dataset_root / "annotations/instances_val2017.json"))
    ids = sorted(annotation.getImgIds())
    if len(ids) != 5000:
        raise ValueError("正式评测需要完整 COCO val2017 5000 张标注.")
    if args.onnx:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(args.onnx), options,
                                       providers=["CPUExecutionProvider"])
    else:
        model = load_model(args.weights).cuda()
        model.model[-1].forward = types.MethodType(raw_forward, model.model[-1])
    predictions = []
    for image_id in tqdm(ids, desc="ONNX 精度" if args.onnx else "PyTorch 精度"):
        record = annotation.loadImgs([image_id])[0]
        tensor, geometry = preprocess(cv2.imread(str(
            args.dataset_root / "images" / record["file_name"])))
        if args.onnx:
            outputs = session.run(OUTPUT_NAMES, {"images": tensor})
        else:
            with torch.no_grad():
                outputs = [value.cpu().numpy() for value in
                           model(torch.from_numpy(tensor).cuda())]
        boxes, scores, classes = postprocess(outputs, geometry)
        for box, score, category in zip(boxes, scores, classes):
            predictions.append({"image_id": image_id,
                                "category_id": COCO_IDS[int(category)],
                                "bbox": [float(box[0]), float(box[1]),
                                         float(box[2] - box[0]), float(box[3] - box[1])],
                                "score": float(score)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output.with_name("predictions.json")
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    if not predictions:
        raise RuntimeError("全量预测为空,请检查模型输出.")
    evaluator = COCOeval(annotation, annotation.loadRes(str(predictions_path)), "bbox")
    evaluator.params.imgIds = ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    summary = {"model": "yolov8n", "backend": "onnx" if args.onnx else "pytorch",
               "samples": len(ids), "map_50_95": float(evaluator.stats[0]),
               "map_50": float(evaluator.stats[1]),
               "protocol": {"confidence": 0.001, "nms_iou": 0.6,
                            "max_det": 300, "class_selection": "single_best",
                            "input": [1, 3, 640, 640]}}
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"[OK] 精度汇总: {args.output}", flush=True)


if __name__ == "__main__":
    main()
