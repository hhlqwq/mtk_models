"""在编译主机运行 YOLOv5s ONNX FP32 COCO 全量精度评测."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from torchvision.ops import batched_nms

COCO_91_CLASSES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
HEAD_SIZES = [80, 40, 20]
ANCHORS = np.array([
    [[1.25, 1.625], [2.0, 3.75], [4.125, 2.875]],
    [[1.875, 3.8125], [3.875, 2.8125], [3.6875, 7.4375]],
    [[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]],
], dtype=np.float32)
STRIDES = [8.0, 16.0, 32.0]


def letterbox(image: np.ndarray, image_size: int) -> tuple:
    """按 YOLOv5 规则缩放并填充图片, 返回画布与仿射元数据."""
    height, width = image.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height))
    canvas = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas, scale, left, top, (height, width)


def decode_heads(heads: list, confidence: float, iou_threshold: float,
                 max_det: int) -> torch.Tensor:
    """解码 3 个检测头并做 NMS, 返回 (N,6) 的 xyxy+score+cls."""
    device = heads[0].device
    predictions = []
    for index, head in enumerate(heads):
        _, _, height, width = head.shape
        values = head.reshape(1, 3, 85, height, width).permute(
            0, 1, 3, 4, 2)
        activated = torch.sigmoid(values)
        grid_x = torch.arange(width, dtype=torch.float32, device=device)
        grid_y = torch.arange(height, dtype=torch.float32, device=device)
        grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing="xy"),
                           -1)[None, None] - 0.5
        xy = (activated[..., :2] * 2.0 + grid) * STRIDES[index]
        anchor_grid = torch.tensor(ANCHORS[index] * STRIDES[index],
                                   dtype=torch.float32,
                                   device=device)[None, :, None, None]
        wh = (activated[..., 2:4] * 2.0) ** 2 * anchor_grid
        predictions.append(torch.cat((xy, wh, activated[..., 4:]),
                                     -1).reshape(-1, 85))
    output = torch.cat(predictions, 0)
    objectness = output[:, 4]
    class_probabilities, classes = output[:, 5:].max(1)
    scores = objectness * class_probabilities
    selected = (scores >= confidence).nonzero().flatten()
    if selected.numel() == 0:
        return torch.empty((0, 6))
    boxes = output[selected, :4].clone()
    boxes[:, :2] -= boxes[:, 2:] / 2.0
    boxes[:, 2:] += boxes[:, :2]
    scores = scores[selected]
    classes = classes[selected]
    kept = batched_nms(boxes, scores, classes, iou_threshold)[:max_det]
    return torch.cat((boxes[kept], scores[kept, None],
                      classes[kept, None].float()), 1)


def rescale_to_original(boxes: torch.Tensor, meta: dict,
                        image_size: int) -> np.ndarray:
    """把画布坐标检测框映射回原始图片尺寸并裁剪."""
    result = boxes.clone().cpu().numpy()
    result[:, [0, 2]] = (result[:, [0, 2]] - meta["left"]) / meta["scale"]
    result[:, [1, 3]] = (result[:, [1, 3]] - meta["top"]) / meta["scale"]
    height, width = meta["original_shape"]
    result[:, [0, 2]] = result[:, [0, 2]].clip(0, width)
    result[:, [1, 3]] = result[:, [1, 3]].clip(0, height)
    return result


def build_fp32_infer(args: argparse.Namespace):
    """构造 ONNX Runtime 推理函数,仅使用 ONNX 模型进行精度评测."""
    import onnxruntime

    providers = [provider for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if provider in onnxruntime.get_available_providers()]
    session = onnxruntime.InferenceSession(str(args.onnx),
                                           providers=providers)
    print(f"[ONNX] 执行后端: {session.get_providers()}", flush=True)

    input_name = session.get_inputs()[0].name
    order = {80: 0, 40: 1, 20: 2}

    def infer(canvas: np.ndarray) -> list:
        """FP32 ONNX Runtime 推理, 输出按 stride 顺序重排."""
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        outputs = session.run(None, {input_name: tensor})
        heads = [None, None, None]
        for value in outputs:
            if value.ndim != 4 or value.shape[-1] not in order:
                raise ValueError(f"ONNX 原始检测头形状异常: {value.shape}")
            if heads[order[value.shape[-1]]] is not None:
                raise ValueError("ONNX 检测头重复.")
            heads[order[value.shape[-1]]] = torch.from_numpy(value)
        return heads

    return infer


def evaluate_onnx_accuracy(args: argparse.Namespace) -> None:
    """在编译主机评测 ONNX 全量精度,仅保存核心 mAP 汇总."""
    import contextlib
    import io
    import math
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        annotation = COCO(str(args.ann))
    image_ids = sorted(annotation.getImgIds())
    paths = sorted(args.images_dir.glob("*.jpg"))
    if len(image_ids) != 5000 or len(paths) != 5000 or {
            int(path.stem) for path in paths} != set(image_ids):
        raise ValueError("图片与标注必须覆盖同一份 COCO val2017 全量 5000 张图片.")
    infer = build_fp32_infer(args)
    records = []
    for path in tqdm.tqdm(paths, desc="ONNX FP32 全量精度", unit="img"):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"无法读取图片: {path}")
        canvas, scale, left, top, shape = letterbox(image, args.image_size)
        heads = infer(canvas)
        if any(head is None for head in heads) or any(
                tuple(head.shape) != (1, 255, size, size)
                for head, size in zip(heads, HEAD_SIZES)):
            raise ValueError("ONNX 必须输出三个原始检测头: 80x80,40x40,20x20.")
        boxes = decode_heads(heads, args.confidence, args.iou, args.max_det)
        boxes = rescale_to_original(
            boxes, {"scale": scale, "left": left, "top": top,
                    "original_shape": shape}, args.image_size)
        for x1, y1, x2, y2, score, class_id in boxes:
            records.append({
                "image_id": int(path.stem),
                "category_id": COCO_91_CLASSES[int(class_id)],
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })
    print("[ONNX] 推理完成,计算 mAP@0.5:0.95.", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        if records:
            prediction = annotation.loadRes(records)
        else:
            prediction = COCO()
            prediction.dataset = {
                "images": annotation.dataset["images"],
                "categories": annotation.dataset["categories"], "annotations": []}
            prediction.createIndex()
        evaluator = COCOeval(annotation, prediction, "bbox")
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    value = float(evaluator.stats[0])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("ONNX mAP 无效.")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps({
        "status": "complete", "backend": "ONNX FP32", "images": len(image_ids),
        "map_50_95": value,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ONNX] mAP@0.5:0.95: {value:.6f}", flush=True)


def parse_args() -> argparse.Namespace:
    """解析 ONNX 模型、全量数据集和汇总输出路径."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--ann", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_onnx_accuracy(parse_args())
