"""ViT-Base Patch16 224 精度评测: FP32 ONNX 基线与 MTK NPU INT8 对齐分析.

阶段 (均在容器内执行, 板端推理由 deploy/accuracy_eval.sh 驱动):

- ``prepare``: 生成 NPU INT8 输入 bin + FP32 ONNX logits npy + 清单.
- ``compare``: 解码 NPU 原生 logits, 与 FP32 基线比较 top-1/top-5 一致率和
  logits 误差; 若提供 labels 同时报告双方绝对 Top-1/Top-5.
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import tqdm


def preprocess(image: np.ndarray, crop_size: int = 224,
               resize_size: int = 256) -> np.ndarray:
    """ImageNet 标准评估几何预处理, 返回 NCHW FP32 [0,1].

    Qualcomm 导出的 ONNX 已在图内完成 mean/std 归一化 (首节点 Sub/Div),
    外部输入必须是 rgb/255 的 [0,1] 范围, 不允许再次归一化.
    """
    height, width = image.shape[:2]
    scale = resize_size / min(height, width)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)),
                         interpolation=cv2.INTER_CUBIC)
    top = (resized.shape[0] - crop_size) // 2
    left = (resized.shape[1] - crop_size) // 2
    crop = resized[top:top + crop_size, left:left + crop_size]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    scaled = rgb.astype(np.float32) / 255.0
    return scaled.transpose(2, 0, 1)[np.newaxis].copy()


def list_images(images_dir: Path) -> list:
    """按文件名排序列出评测图片."""
    return sorted(p for p in images_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPEG"})


def imagenet_index_from_name(image_name: str) -> int:
    """从 ILSVRC2012 验证图片名解析 0-based 图片序号."""
    match = re.fullmatch(r"ILSVRC2012_val_(\d{8})\.(?:JPEG|jpeg|jpg)",
                         image_name)
    if match is None:
        raise ValueError(f"ImageNet 验证图片名格式异常: {image_name}")
    image_number = int(match.group(1))
    if image_number < 1 or image_number > 50000:
        raise ValueError(f"ImageNet 验证图片编号越界: {image_name}")
    return image_number - 1


def load_labels(path: Path, required_indices: list[int]) -> dict[int, int]:
    """加载 ImageNet val ground truth, 并校验本次评测所需序号.

    此处只接受已经映射到模型输出顺序的 0-based 类 id.单列格式按图片排序
    逐行对应；两列格式为 "1-based 图片序号 0-based 类 id".原始 ILSVRC
    ground truth 的 1-based synset id 必须先结合 devkit meta.mat 完成映射.
    标签文件可覆盖完整 50,000 张, 本次评测只要求所需图片序号全部存在.
    """
    labels: dict[int, int] = {}
    for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines()):
        parts = line.split()
        if not parts:
            continue
        index = int(parts[0]) - 1 if len(parts) > 1 else line_no
        class_id = int(parts[-1])
        if index < 0 or index >= 50000 or class_id < 0 or class_id >= 1000:
            raise ValueError(
                f"标签越界: line={line_no + 1}, index={index}, "
                f"class_id={class_id}")
        if index in labels:
            raise ValueError(f"标签图片序号重复: {index}")
        labels[index] = class_id
    missing = [index for index in required_indices if index not in labels]
    if missing:
        preview = ", ".join(str(index) for index in missing[:10])
        raise ValueError(
            f"缺少本次评测所需标签: count={len(missing)}, indices={preview}")
    return labels


def load_manifest(path: Path, start: int, count: int) -> list[dict]:
    """加载清单, 并严格校验序号、文件名和顺序."""
    manifest = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(manifest) != count:
        raise ValueError(
            f"清单数量错误: 需要 {count}, 实际 {len(manifest)}")
    expected_indices = list(range(start, start + count))
    actual_indices = [record["index"] for record in manifest]
    if actual_indices != expected_indices:
        raise ValueError(
            f"清单图片序号不连续或顺序异常: start={start}, count={count}")
    for record in manifest:
        index = int(record["index"])
        image_name = str(record["image"])
        if imagenet_index_from_name(image_name) != index:
            raise ValueError(
                f"清单图片名与序号不一致: index={index}, image={image_name}")
        expected_prefix = f"{index:06d}_"
        if not str(record["stem"]).startswith(expected_prefix):
            raise ValueError(
                f"清单 stem 与序号不一致: index={index}, stem={record['stem']}")
    return manifest


def topk_from_logits(logits: np.ndarray, k: int) -> np.ndarray:
    """返回 logits 的 top-k 类 id."""
    return np.argpartition(logits, -k)[-k:][np.argsort(
        -logits[np.argpartition(logits, -k)[-k:]])]


def load_npu_logit(path: Path, out_size: int, scale: float,
                   zero_point: int) -> np.ndarray:
    """读取并反量化 NPU 原生 logits (兼容 16 对齐行 padding)."""
    raw = np.fromfile(path, dtype=np.int8)
    pad = (out_size + 15) // 16 * 16
    if raw.size == pad:
        raw = raw[:out_size]
    elif raw.size != out_size:
        raise ValueError(f"logit 大小异常: {path} {raw.size}")
    return (raw.astype(np.float32) - zero_point) * scale


def stage_prepare(args: argparse.Namespace) -> None:
    """生成 NPU 输入 bin、FP32 基线 logits 与清单."""
    import mtk_converter
    import onnxruntime

    image_paths = list_images(args.images_dir)[args.start:args.start + args.count]
    if len(image_paths) != args.count:
        raise ValueError(
            f"评测图片不足: start={args.start}, 需要 {args.count}, "
            f"实际 {len(image_paths)}")
    parser = mtk_converter.TFLiteParser(str(args.tflite))
    input_detail = parser.get_input_tensor_details()[0]
    if list(input_detail["shape"]) != [1, 3, 224, 224]:
        raise ValueError(f"TFLite 输入 shape 异常: {input_detail['shape']}")
    q_scale = float(input_detail["quantization"]["scales"][0])
    q_zero = int(input_detail["quantization"]["zero_points"][0])
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.onnx_provider == "cuda"
                 else ["CPUExecutionProvider"])
    session = onnxruntime.InferenceSession(str(args.onnx),
                                           providers=providers)
    if (args.onnx_provider == "cuda" and
            "CUDAExecutionProvider" not in session.get_providers()):
        raise RuntimeError(
            "ONNX Runtime 未启用 CUDAExecutionProvider,拒绝静默回退 CPU.")
    if args.onnx_provider == "cpu":
        print("[INFO] FP32 ONNX 显式使用 CPUExecutionProvider.")
    input_name = session.get_inputs()[0].name
    args.bins_dir.mkdir(parents=True, exist_ok=True)
    args.logits_dir.mkdir(parents=True, exist_ok=True)
    manifest_lines = []
    for position, image_path in enumerate(tqdm.tqdm(
            image_paths, desc="ViT 评测准备", unit="img")):
        global_position = args.start + position
        image_index = imagenet_index_from_name(image_path.name)
        if image_index != global_position:
            raise ValueError(
                f"图片排序与官方编号不一致: index={global_position}, "
                f"image={image_path.name}")
        stem = f"{global_position:06d}_{image_path.stem}"
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        fp32 = preprocess(image)
        quantized = np.clip(np.round(fp32 / q_scale) + q_zero, -128,
                            127).astype(np.int8)
        quantized.tofile(args.bins_dir / f"{stem}.bin")
        logits = session.run(None, {input_name: fp32})[0].reshape(-1)
        logits.astype(np.float32).tofile(
            args.logits_dir / f"{stem}.f32")
        manifest_lines.append(json.dumps({
            "index": global_position,
            "image": image_path.name,
            "stem": stem,
        }, ensure_ascii=False))
    with args.manifest.open("a", encoding="utf-8", newline="\n") as file:
        for line in manifest_lines:
            file.write(f"{line}\n")
    print(f"[OK] prepare 完成: {len(image_paths)} 张.")


def stage_compare(args: argparse.Namespace) -> None:
    """NPU logits 与 FP32 基线对齐分析, 可选报告绝对精度."""
    import mtk_converter

    parser = mtk_converter.TFLiteParser(str(args.tflite))
    output_details = parser.get_output_tensor_details()
    if len(output_details) != 1 or list(output_details[0]["shape"]) != [1, 1000]:
        raise ValueError(f"TFLite 输出结构异常: {output_details}")
    out_detail = output_details[0]
    out_scale = float(out_detail["quantization"]["scales"][0])
    out_zero = int(out_detail["quantization"]["zero_points"][0])
    out_size = int(np.prod(out_detail["shape"]))
    top1_match = 0
    top5_overlap = 0.0
    max_abs = 0.0
    total_sq = 0.0
    total_ref = 0.0
    npu_top1_correct = 0
    fp32_top1_correct = 0
    npu_top5_correct = 0
    fp32_top5_correct = 0
    holdout_count = 0
    npu_holdout_top1_correct = 0
    fp32_holdout_top1_correct = 0
    npu_holdout_top5_correct = 0
    fp32_holdout_top5_correct = 0
    manifest = load_manifest(args.manifest, args.start, args.count)
    required_indices = [int(record["index"]) for record in manifest]
    labels = (load_labels(args.labels, required_indices)
              if args.labels else None)
    exclude_stop = (args.exclude_accuracy_start + args.exclude_accuracy_count)
    output_count = len(list(args.npu_dir.glob("*_0.bin")))
    if output_count != args.count:
        raise ValueError(
            f"NPU 输出数量错误: 需要 {args.count}, 实际 {output_count}")
    count = 0
    for record in manifest:
        stem = record["stem"]
        ref_path = args.logits_dir / f"{stem}.f32"
        dla_out = args.npu_dir / f"{stem}_0.bin"
        if not ref_path.is_file() or ref_path.stat().st_size != out_size * 4:
            raise ValueError(f"FP32 logits 缺失或大小错误: {ref_path}")
        if not dla_out.is_file() or dla_out.stat().st_size == 0:
            raise ValueError(f"NPU 输出缺失或为空: {dla_out}")
        ref = ref_path.read_bytes()
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
            index = int(record["index"])
            truth = labels[index]
            npu_top1_correct += int(npu_top[0] == truth)
            fp32_top1_correct += int(ref_top[0] == truth)
            npu_top5_correct += int(truth in npu_top.tolist())
            fp32_top5_correct += int(truth in ref_top.tolist())
            if not args.exclude_accuracy_start <= index < exclude_stop:
                holdout_count += 1
                npu_holdout_top1_correct += int(npu_top[0] == truth)
                fp32_holdout_top1_correct += int(ref_top[0] == truth)
                npu_holdout_top5_correct += int(truth in npu_top.tolist())
                fp32_holdout_top5_correct += int(truth in ref_top.tolist())
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
            "labeled_samples": count,
            "npu_top1": round(npu_top1_correct / max(count, 1), 4),
            "npu_top5": round(npu_top5_correct / max(count, 1), 4),
            "fp32_top1": round(fp32_top1_correct / max(count, 1), 4),
            "fp32_top5": round(fp32_top5_correct / max(count, 1), 4),
            "accuracy_excluded_range": {
                "start_0based": args.exclude_accuracy_start,
                "count": args.exclude_accuracy_count,
            },
            "holdout_samples": holdout_count,
            "npu_holdout_top1": round(
                npu_holdout_top1_correct / max(holdout_count, 1), 4),
            "npu_holdout_top5": round(
                npu_holdout_top5_correct / max(holdout_count, 1), 4),
            "fp32_holdout_top1": round(
                fp32_holdout_top1_correct / max(holdout_count, 1), 4),
            "fp32_holdout_top5": round(
                fp32_holdout_top5_correct / max(holdout_count, 1), 4),
        })
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[OK] summary -> {args.summary}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
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
    parser.add_argument("--onnx-provider", choices=["cpu", "cuda"],
                        default="cuda")
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--bins-dir", type=Path, default=None)
    parser.add_argument("--logits-dir", type=Path, default=None)
    parser.add_argument("--npu-dir", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--exclude-accuracy-start", type=int, default=1000,
                        help="绝对精度独立指标排除区间的 0-based 起点.")
    parser.add_argument("--exclude-accuracy-count", type=int, default=100,
                        help="绝对精度独立指标排除区间的样本数.")
    args = parser.parse_args()
    if args.start < 0 or args.count <= 0:
        parser.error("--start 必须大于等于 0, --count 必须大于 0.")
    if args.exclude_accuracy_start < 0 or args.exclude_accuracy_count < 0:
        parser.error("精度排除区间参数不能为负数.")
    args.bins_dir = args.bins_dir or args.work_dir / "npu_bins"
    args.logits_dir = args.logits_dir or args.work_dir / "fp32_logits"
    args.npu_dir = args.npu_dir or args.work_dir / "npu_outputs"
    args.summary = args.summary or args.work_dir / "agreement_summary.json"
    args.manifest = args.manifest or args.work_dir / "manifest.jsonl"
    return args


if __name__ == "__main__":
    parsed = parse_args()
    {"prepare": stage_prepare, "compare": stage_compare}[parsed.stage](parsed)
