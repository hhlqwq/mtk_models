"""ViT-Base Patch16 224 精度评测: FP32 ONNX 基线与 MTK NPU INT8 对齐分析。

阶段 (均在容器内执行, 板端推理由 deploy/accuracy_eval.sh 驱动):

- ``prepare``: 生成 NPU INT8 输入 bin + FP32 ONNX logits npy + 清单。
- ``compare``: 解码 NPU 原生 logits, 与 FP32 基线比较 top-1/top-5 一致率和
  logits 误差; 若提供 labels 同时报告双方绝对 Top-1/Top-5。
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tqdm

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: np.ndarray, crop_size: int = 224,
               resize_size: int = 256) -> np.ndarray:
    """ImageNet 标准评估预处理, 返回 NCHW FP32。"""
    height, width = image.shape[:2]
    scale = resize_size / min(height, width)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)),
                         interpolation=cv2.INTER_CUBIC)
    top = (resized.shape[0] - crop_size) // 2
    left = (resized.shape[1] - crop_size) // 2
    crop = resized[top:top + crop_size, left:left + crop_size]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return normalized.transpose(2, 0, 1)[np.newaxis].copy()


def list_images(images_dir: Path) -> list:
    """按文件名排序列出评测图片。"""
    return sorted(p for p in images_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPEG"})


def load_labels(path: Path, count: int) -> dict:
    """加载 ImageNet val ground truth (行号 -> 0-based 类 id)。

    支持 ILSVRC devkit 的 ILSVRC2012_validation_ground_truth.txt (1-based WNID
    顺序表需要 meta.mat, 此处接受已映射为类序号的两列/单列文本:
    每行 "序号 类id" 或仅 "类id", 类 id 为 0-based)。
    """
    labels = {}
    for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines()):
        parts = line.split()
        if not parts:
            continue
        index = int(parts[0]) - 1 if len(parts) > 1 else line_no
        labels[index] = int(parts[-1])
    return labels


def topk_from_logits(logits: np.ndarray, k: int) -> np.ndarray:
    """返回 logits 的 top-k 类 id。"""
    return np.argpartition(logits, -k)[-k:][np.argsort(
        -logits[np.argpartition(logits, -k)[-k:]])]


def load_npu_logit(path: Path, out_size: int, scale: float,
                   zero_point: int) -> np.ndarray:
    """读取并反量化 NPU 原生 logits (兼容 16 对齐行 padding)。"""
    raw = np.fromfile(path, dtype=np.int8)
    pad = (out_size + 15) // 16 * 16
    if raw.size == pad:
        raw = raw[:out_size]
    elif raw.size != out_size:
        raise ValueError(f"logit 大小异常: {path} {raw.size}")
    return (raw.astype(np.float32) - zero_point) * scale


def stage_prepare(args: argparse.Namespace) -> None:
    """生成 NPU 输入 bin、FP32 基线 logits 与清单。"""
    import mtk_converter
    import onnxruntime

    image_paths = list_images(args.images_dir)[args.start:args.start + args.count]
    parser = mtk_converter.TFLiteParser(str(args.tflite))
    input_detail = parser.get_input_tensor_details()[0]
    q_scale = input_detail["quantization"]["scales"][0]
    q_zero = input_detail["quantization"]["zero_points"][0]
    session = onnxruntime.InferenceSession(
        str(args.onnx), providers=["CUDAExecutionProvider",
                                   "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    args.bins_dir.mkdir(parents=True, exist_ok=True)
    args.logits_dir.mkdir(parents=True, exist_ok=True)
    for position, image_path in enumerate(tqdm.tqdm(
            image_paths, desc="ViT 评测准备", unit="img")):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        fp32 = preprocess(image)
        quantized = np.clip(np.round(fp32 / q_scale) + q_zero, -128,
                            127).astype(np.int8)
        quantized.tofile(args.bins_dir / f"{position:06d}_{image_path.stem}.bin")
        logits = session.run(None, {input_name: fp32})[0].reshape(-1)
        logits.astype(np.float32).tofile(
            args.logits_dir / f"{position:06d}_{image_path.stem}.npy")
    print(f"[OK] prepare 完成: {len(image_paths)} 张。")


def stage_compare(args: argparse.Namespace) -> None:
    """NPU logits 与 FP32 基线对齐分析, 可选报告绝对精度。"""
    import mtk_converter

    parser = mtk_converter.TFLiteParser(str(args.tflite))
    out_detail = parser.get_output_tensor_details()[0]
    out_scale = out_detail["quantization"]["scales"][0]
    out_zero = out_detail["quantization"]["zero_points"][0]
    out_size = int(np.prod(out_detail["shape"]))
    labels = load_labels(args.labels, args.count) if args.labels else None

    top1_match = 0
    top5_overlap = 0.0
    max_abs = 0.0
    total_sq = 0.0
    total_ref = 0.0
    npu_top1_correct = 0
    fp32_top1_correct = 0
    npu_top5_correct = 0
    fp32_top5_correct = 0
    count = 0
    for npy in sorted(args.logits_dir.glob("*.npy")):
        stem = npy.name[:-len(".npy")]
        dla_out = args.npu_dir / f"{stem}_0.bin"
        if not dla_out.exists():
            continue
        ref = npy.read_bytes()
        ref_logits = np.frombuffer(ref, dtype=np.float32)
        npu_logits = load_npu_logit(dla_out, out_size, out_scale, out_zero)
        ref_top = topk_from_logits(ref_logits, 5)
        npu_top = topk_from_logits(npu_logits, 5)
        top1_match += int(ref_top[0] == npu_top[0])
        top5_overlap += len(set(ref_top.tolist()) & set(npu_top.tolist())) / 5.0
        diff = npu_logits - ref_logits
        max_abs = max(max_abs, float(np.abs(diff).max()))
        total_sq += float((diff ** 2).sum())
        total_ref += float((ref_logits ** 2).sum())
        if labels is not None:
            index = int(stem.split("_")[0])
            truth = labels.get(index)
            if truth is not None:
                npu_top1_correct += int(npu_top[0] == truth)
                fp32_top1_correct += int(ref_top[0] == truth)
                npu_top5_correct += int(truth in npu_top.tolist())
                fp32_top5_correct += int(truth in ref_top.tolist())
        count += 1
        if count % 500 == 0:
            print(f"  已对比 {count} 张 ...")
    summary = {
        "samples": count,
        "top1_agreement": round(top1_match / max(count, 1), 4),
        "top5_overlap": round(top5_overlap / max(count, 1), 4),
        "logit_max_abs_err": round(max_abs, 4),
        "logit_rmse": round((total_sq / max(count, 1) /
                             max(out_size, 1)) ** 0.5, 4),
        "logit_relative_frobenius": round(
            (total_sq / max(total_ref, 1e-9)) ** 0.5, 4),
    }
    if labels is not None:
        summary.update({
            "npu_top1": round(npu_top1_correct / max(count, 1), 4),
            "npu_top5": round(npu_top5_correct / max(count, 1), 4),
            "fp32_top1": round(fp32_top1_correct / max(count, 1), 4),
            "fp32_top5": round(fp32_top5_correct / max(count, 1), 4),
        })
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[OK] summary -> {args.summary}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["prepare", "compare"])
    parser.add_argument("--work-dir", type=Path,
                        default=Path("/workspace/.eval/vit_base_patch16_224"))
    parser.add_argument("--images-dir", type=Path,
                        default=Path("/data/users/hailong.he/nas_smb/Datasets/"
                                     "open_source/raw/ILSVRC2012/val"))
    parser.add_argument("--tflite", type=Path,
                        default=Path("/workspace/models/perception/"
                                     "image_classification/"
                                     "vit_base_patch16_224/models/"
                                     "model_int8.tflite"))
    parser.add_argument("--onnx", type=Path,
                        default=Path("/workspace/models/perception/"
                                     "image_classification/"
                                     "vit_base_patch16_224/models/"
                                     "model_fp32.onnx"))
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--bins-dir", type=Path, default=None)
    parser.add_argument("--logits-dir", type=Path, default=None)
    parser.add_argument("--npu-dir", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    args.bins_dir = args.bins_dir or args.work_dir / "npu_bins"
    args.logits_dir = args.logits_dir or args.work_dir / "fp32_logits"
    args.npu_dir = args.npu_dir or args.work_dir / "npu_outputs"
    args.summary = args.summary or args.work_dir / "agreement_summary.json"
    return args


if __name__ == "__main__":
    parsed = parse_args()
    {"prepare": stage_prepare, "compare": stage_compare}[parsed.stage](parsed)
