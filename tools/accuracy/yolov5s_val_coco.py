"""YOLOv5s COCO val2017 三后端统一精度评测 (PyTorch / ONNX / MTK NPU).

三个后端共享同一 letterbox 预处理、解码和 NMS 逻辑, 保证公平对比:

- ``prepare``: 生成板端 NPU 推理所需 INT8 输入 bin 与清单.
- ``decode``: 对某一后端的原始推理结果做解码 + NMS, 输出 COCO 结果 jsonl.
- ``evaluate``: 用 pycocotools 计算 mAP 并写出 summary.
"""

import argparse
import json
import sys
import types
from collections import Counter
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


def load_manifest(path: Path) -> dict:
    """读取输入清单; 不存在时返回空 dict."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    """原子写出输入清单."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest), encoding="utf-8")
    tmp.replace(path)


def stage_prepare(args: argparse.Namespace) -> None:
    """生成 [start, start+count) 范围图片的 INT8 输入 bin 并更新清单."""
    import mtk_converter

    image_paths = sorted(Path(args.images_dir).glob("*.jpg"))
    subset = image_paths[args.start:args.start + args.count]
    if len(subset) != args.count:
        raise ValueError(
            f"评测图片不足: 请求 [{args.start}, {args.start + args.count}), "
            f"实际仅取得 {len(subset)} 张")
    parser = mtk_converter.TFLiteParser(str(args.tflite))
    input_detail = parser.get_input_tensor_details()[0]
    q_scale = input_detail["quantization"]["scales"][0]
    q_zero = input_detail["quantization"]["zero_points"][0]
    manifest = load_manifest(args.manifest)
    args.bins_dir.mkdir(parents=True, exist_ok=True)
    for image_path in tqdm.tqdm(subset, desc="准备 INT8 输入", unit="img"):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        canvas, scale, left, top, original_shape = letterbox(
            image, args.image_size)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        nchw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        quantized = np.clip(np.round(nchw / q_scale) + q_zero, -128,
                            127).astype(np.int8)
        quantized[np.newaxis].tofile(args.bins_dir / f"{image_path.stem}.bin")
        orig_h, orig_w = original_shape
        manifest[image_path.stem] = {
            "scale": float(scale),
            "left": int(left),
            "top": int(top),
            "original_shape": [int(orig_h), int(orig_w)],
        }
    save_manifest(args.manifest, manifest)
    print(f"[OK] prepare 完成, 累计清单 {len(manifest)} 条.")


def rowpadded_to_nchw(buffer: np.ndarray, height: int, width: int,
                      channels: int) -> np.ndarray:
    """按 NCHW 行 stride 16 对齐 (或无 padding) 还原张量."""
    plain = channels * height * width
    pad = (width + 15) // 16 * 16
    padded = channels * height * pad
    if buffer.size == plain:
        return buffer.reshape(1, channels, height, width)
    if buffer.size == padded:
        return buffer.reshape(1, channels, height, pad)[..., :width].copy()
    raise ValueError(
        f"输出大小异常: {buffer.size}, 期望 {plain} 或 {padded}")


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


def append_results(path: Path, image_id: int, boxes: np.ndarray) -> None:
    """按 COCO 结果格式追加 jsonl 记录."""
    with path.open("a", encoding="utf-8") as handle:
        for *xyxy, score, class_id in boxes:
            x1, y1, x2, y2 = xyxy
            handle.write(json.dumps({
                "image_id": image_id,
                "category_id": COCO_91_CLASSES[int(class_id)],
                "bbox": [float(round(x1, 3)), float(round(y1, 3)),
                         float(round(x2 - x1, 3)), float(round(y2 - y1, 3))],
                "score": round(float(score), 5),
            }) + "\n")


def result_image_ids(path: Path) -> set[int]:
    """读取结果 jsonl 中已有预测的 image_id 集合."""
    if not path.exists():
        return set()
    return {json.loads(line)["image_id"]
            for line in path.read_text(encoding="utf-8").splitlines() if line}


def processed_image_ids(path: Path) -> set[int]:
    """读取已完成图片列表,包含没有任何检测结果的图片."""
    if not path.exists():
        return set()
    return {int(line.split(",", maxsplit=1)[0]) for line in path.read_text(
        encoding="utf-8").splitlines() if line}


def processed_record_counts(path: Path) -> dict[int, int]:
    """读取各图片应有的检测记录数."""
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        image_id, count = line.split(",", maxsplit=1)
        result[int(image_id)] = int(count)
    return result


def mark_processed(path: Path, image_id: int, record_count: int) -> None:
    """在结果完整写出后追加图片完成标记和记录数."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{image_id},{record_count}\n")


def stage_decode(args: argparse.Namespace) -> None:
    """对指定后端输出做解码 + NMS, 写 COCO 结果 jsonl."""
    manifest = load_manifest(args.manifest)
    if not manifest:
        raise ValueError(f"输入清单为空: {args.manifest}")
    args.result.parent.mkdir(parents=True, exist_ok=True)
    processed = processed_image_ids(args.done)
    orphan_results = result_image_ids(args.result) - processed
    if orphan_results:
        raise RuntimeError(
            "结果文件包含未完成记录,可能是上次写入中断；请使用新的 "
            f"EVAL_RUN_ID.示例 image_id: {min(orphan_results)}")
    if args.backend == "npu":
        import mtk_converter

        parser = mtk_converter.TFLiteParser(str(args.tflite))
        outputs = parser.get_output_tensor_details()
        output_shapes = [list(output["shape"]) for output in outputs]
        expected_shapes = [[1, 255, size, size] for size in HEAD_SIZES]
        if output_shapes != expected_shapes:
            raise ValueError(
                f"TFLite 输出顺序或形状异常: {output_shapes}, "
                f"期望 {expected_shapes}")
        scales = [o["quantization"]["scales"][0] for o in outputs]
        zeros = [o["quantization"]["zero_points"][0] for o in outputs]
        items = sorted(manifest.items(), key=lambda item: int(item[0]))
        for stem, meta in tqdm.tqdm(items, desc="解码 NPU 输出", unit="img"):
            image_id = int(stem)
            if image_id in processed:
                continue
            heads = []
            for index, size in enumerate(HEAD_SIZES):
                output_path = args.bins_dir / f"{stem}_{index}.bin"
                if not output_path.is_file() or output_path.stat().st_size == 0:
                    raise FileNotFoundError(f"NPU 输出缺失或为空: {output_path}")
                raw = np.fromfile(output_path, dtype=np.int8)
                nchw = rowpadded_to_nchw(raw, size, size, 255)
                values = (nchw.astype(np.float32) - zeros[index]) * scales[index]
                heads.append(torch.from_numpy(values))
            boxes = decode_heads(heads, args.confidence, args.iou,
                                 args.max_det)
            append_results(args.result, image_id, rescale_to_original(
                boxes, meta, args.image_size))
            mark_processed(args.done, image_id, len(boxes))
    else:
        infer = build_fp32_infer(args)
        items = sorted(manifest.items(), key=lambda item: int(item[0]))
        for stem, meta in tqdm.tqdm(items, desc=f"解码 {args.backend}",
                                    unit="img"):
            image_id = int(stem)
            if image_id in processed:
                continue
            image = cv2.imread(str(args.images_dir / f"{stem}.jpg"))
            if image is None:
                raise ValueError(f"缺少图片: {stem}.jpg")
            canvas, _, _, _, _ = letterbox(image, args.image_size)
            heads = infer(canvas)
            boxes = decode_heads(heads, args.confidence, args.iou,
                                 args.max_det)
            append_results(args.result, image_id, rescale_to_original(
                boxes, meta, args.image_size))
            mark_processed(args.done, image_id, len(boxes))
    print(f"[OK] decode({args.backend}) 完成 -> {args.result}")


def build_fp32_infer(args: argparse.Namespace):
    """构造返回 3 个原始检测头的 FP32 推理函数 (torch 或 onnx)."""
    if args.backend == "torch":
        sys.path.insert(0, str(args.source_dir))
        from models.experimental import attempt_load

        model = attempt_load(str(args.weights), device="cuda")
        detect = model.model[-1]

        def detect_forward(self, x):
            """Detect 仅保留 3 个 conv 输出, 与部署图一致."""
            return [self.m[i](x[i]) for i in range(self.nl)]

        detect.forward = types.MethodType(detect_forward, detect)
        model.eval().cuda()

        def infer(canvas: np.ndarray) -> list:
            """FP32 PyTorch 推理."""
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(
                rgb.transpose(2, 0, 1).astype(np.float32) / 255.0).cuda()[None]
            with torch.no_grad():
                return model(tensor)

        return infer
    import onnxruntime

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = onnxruntime.InferenceSession(str(args.onnx),
                                           providers=providers)
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            "ONNX Runtime 未启用 CUDAExecutionProvider,拒绝静默回退 CPU.")
    input_name = session.get_inputs()[0].name
    order = {80: 0, 40: 1, 20: 2}

    def infer(canvas: np.ndarray) -> list:
        """FP32 ONNX Runtime 推理, 输出按 stride 顺序重排."""
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        outputs = session.run(None, {input_name: tensor})
        heads = [None, None, None]
        for value in outputs:
            heads[order[value.shape[-1]]] = torch.from_numpy(value)
        return heads

    return infer


def stage_evaluate(args: argparse.Namespace) -> None:
    """用 pycocotools 计算 mAP 并写出 summary."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    manifest = load_manifest(args.manifest)
    expected_ids = {int(image_id) for image_id in manifest}
    processed_ids = processed_image_ids(args.done)
    if not expected_ids:
        raise ValueError(f"输入清单为空: {args.manifest}")
    if args.expected_images and len(expected_ids) != args.expected_images:
        raise RuntimeError(
            f"清单图片数异常: 期望 {args.expected_images}, "
            f"实际 {len(expected_ids)}")
    if processed_ids != expected_ids:
        missing = sorted(expected_ids - processed_ids)
        unexpected = sorted(processed_ids - expected_ids)
        raise RuntimeError(
            "评测覆盖不完整: "
            f"期望 {len(expected_ids)}, 已完成 {len(processed_ids)}, "
            f"缺少 {missing[:3]}, 多余 {unexpected[:3]}")
    records = [json.loads(line) for line in
               args.result.read_text(encoding="utf-8").splitlines() if line]
    expected_record_counts = processed_record_counts(args.done)
    record_counter = Counter(record["image_id"] for record in records)
    actual_record_counts = {
        image_id: record_counter.get(image_id, 0) for image_id in expected_ids
    }
    if actual_record_counts != expected_record_counts:
        raise RuntimeError(
            "结果记录数与完成标记不一致,可能发生中断或文件损坏；"
            "请使用新的 EVAL_RUN_ID.")
    unexpected_results = result_image_ids(args.result) - expected_ids
    if unexpected_results:
        raise RuntimeError(
            f"结果包含清单外 image_id: {sorted(unexpected_results)[:3]}")
    print(f"[INFO] {args.result.name}: {len(records)} 条检测结果")
    annotation = COCO(str(args.ann))
    prediction = annotation.loadRes(records)
    evaluator = COCOeval(annotation, prediction, "bbox")
    # 必须评测清单中的全部图片,包括没有任何预测结果的图片.
    evaluator.params.imgIds = sorted(expected_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ["AP50:95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    summary = {name: round(float(value), 4)
               for name, value in zip(names, evaluator.stats)}
    summary["backend"] = args.backend
    summary["evaluated_images"] = len(expected_ids)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] summary -> {args.summary}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["prepare", "decode", "evaluate"])
    parser.add_argument("--images-dir", type=Path,
                        default=Path("/data/users/hailong.he/nas_smb/Datasets/"
                                     "open_source/raw/coco/coco_val2017/images"))
    parser.add_argument("--ann", type=Path,
                        default=Path("/data/users/hailong.he/nas_smb/Datasets/"
                                     "open_source/raw/coco/coco_val2017/"
                                     "annotations/instances_val2017.json"))
    parser.add_argument("--tflite", type=Path,
                        default=Path("/workspace/models/perception/"
                                     "object_detection/yolov5s/models/"
                                     "model_int8.tflite"))
    parser.add_argument("--onnx", type=Path,
                        default=Path("/workspace/models/perception/"
                                     "object_detection/yolov5s/models/"
                                     "model_fp32.onnx"))
    parser.add_argument("--weights", type=Path,
                        default=Path("/workspace/models/perception/"
                                     "object_detection/yolov5s/models/"
                                     "yolov5s.pt"))
    parser.add_argument("--source-dir", type=Path,
                        default=Path("/workspace/models/perception/"
                                     "object_detection/yolov5s/original/yolov5"))
    parser.add_argument("--work-dir", type=Path,
                        default=Path("/workspace/.eval/yolov5s"))
    parser.add_argument("--bins-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--done", type=Path, default=None)
    parser.add_argument("--backend", default="npu",
                        choices=["npu", "torch", "onnx"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--expected-images", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()
    args.bins_dir = args.bins_dir or args.work_dir / f"{args.backend}_bins"
    args.manifest = args.manifest or args.work_dir / "manifest.json"
    args.result = args.result or args.work_dir / f"{args.backend}_results.jsonl"
    args.summary = args.summary or args.work_dir / f"{args.backend}_summary.json"
    args.done = args.done or args.work_dir / f"{args.backend}_done_ids.txt"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    return args


STAGES = {"prepare": stage_prepare, "decode": stage_decode,
          "evaluate": stage_evaluate}

if __name__ == "__main__":
    parsed = parse_args()
    STAGES[parsed.stage](parsed)
