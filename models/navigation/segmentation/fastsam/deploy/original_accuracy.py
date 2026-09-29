"""使用官方 FastSAM-s PyTorch 权重评测 COCO 类别无关分割精度。"""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from ultralytics.yolo.utils import ops

from fastsam_utils import preprocess
from full_accuracy_board import make_class_agnostic_ground_truth


def sha256_file(path: Path) -> str:
    """逐块计算模型或标注的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """以固定方形输入运行原始模型并保存完整 COCO AP 证据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path,
                        default=Path(__file__).with_name("accuracy_protocol.json"))
    parser.add_argument("--limit", type=int, default=0,
                        help="诊断时仅评测排序后的前 N 图；0 表示完整验证集。")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if protocol["input_size"] != 640 or protocol["images"] != 5000:
        raise ValueError("FastSAM 原始端需要 640 输入与完整 5,000 图协议。")
    images = sorted(args.images.glob("*.jpg"))
    ground_truth = make_class_agnostic_ground_truth(args.annotations)
    expected = set(ground_truth.getImgIds())
    if len(images) != protocol["images"] or {
            int(image.stem) for image in images} != expected:
        raise ValueError("COCO val2017 图片或标注不完整。")
    if args.limit < 0 or args.limit > len(images):
        raise ValueError("--limit 必须在 0 到 5,000 之间。")
    if args.limit:
        images = images[:args.limit]
    args.output_dir.mkdir(parents=True)
    model = YOLO(str(args.weights)).model.cuda().eval()
    predictions = []
    with torch.no_grad():
        for index, image_path in enumerate(images, 1):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"图片无法解码: {image_path}。")
            tensor, _ = preprocess(image)
            outputs = model(torch.from_numpy(tensor).cuda())
            detections = ops.non_max_suppression(
                outputs[0], protocol["confidence"], protocol["nms_iou"],
                agnostic=True, max_det=protocol["max_detections"], nc=1)[0]
            if len(detections):
                detections[:, :4] = ops.scale_boxes(
                    (640, 640), detections[:, :4], image.shape)
                masks = ops.process_mask_native(
                    outputs[1][-1][0], detections[:, 6:], detections[:, :4],
                    image.shape[:2]).cpu().numpy() > 0.5
                scores = detections[:, 4].cpu().numpy()
                for score, mask in zip(scores, masks):
                    encoded = mask_utils.encode(
                        np.asfortranarray(mask.astype(np.uint8)))
                    encoded["counts"] = encoded["counts"].decode("ascii")
                    predictions.append({
                        "image_id": int(image_path.stem), "category_id": 1,
                        "segmentation": encoded, "score": float(score),
                    })
            if index % 100 == 0 or index == len(images):
                print(f"[PROGRESS] FastSAM 原始端 {index}/{len(images)} 张。",
                      flush=True)
    prediction_path = args.output_dir / "coco_segm_predictions.json"
    prediction_path.write_text(json.dumps(predictions, ensure_ascii=False),
                               encoding="utf-8")
    detection = ground_truth.loadRes(str(prediction_path))
    evaluator = COCOeval(ground_truth, detection, "segm")
    evaluator.params.imgIds = [int(image.stem) for image in images]
    evaluator.params.catIds = [1]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    (args.output_dir / "cocoeval.log").write_text(output.getvalue(),
                                                    encoding="utf-8")
    report = {
        "status": "complete" if not args.limit else "diagnostic_subset",
        "backend": "pytorch_fp32_cuda",
        "dataset": protocol["dataset"],
        "protocol": protocol,
        "images": len(images),
        "predictions": len(predictions),
        "ap50_95": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "weights_sha256": sha256_file(args.weights),
        "annotations_sha256": sha256_file(args.annotations),
        "predictions_sha256": sha256_file(prediction_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[RESULT] FastSAM 原始端 AP50:95 {report['ap50_95']:.6f}。",
          flush=True)


if __name__ == "__main__":
    main()
