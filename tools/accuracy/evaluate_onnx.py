"""在编译主机评测各模型 ONNX 精度,复用板端的数据与后处理协议."""

import argparse
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import onnxruntime as ort
import tqdm


ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "vit_base_patch16_224": "perception/image_classification/vit_base_patch16_224",
    "rtmpose_body2d": "interaction/pose_detection/rtmpose_body2d",
    "mobilefacenet": "interaction/face_recognition/mobilefacenet",
    "depth_anything_v2_small": "navigation/single_camera_depth/depth_anything_v2_small",
    "fastsam": "navigation/segmentation/fastsam",
    "yoloworld_xl": "perception/object_detection/yoloworld_xl",
    "whisper_tiny": "audio/stt/whisper_tiny",
}


def load_module(name: str, path: Path):
    """按明确路径加载已有处理函数,避免不同模型同名模块冲突."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_session(path: Path) -> ort.InferenceSession:
    """加载 ONNX 浮点基准,允许使用编译主机 CPU 或 CUDA."""
    providers = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if name in ort.get_available_providers()]
    session = ort.InferenceSession(str(path), providers=providers)
    print(f"[ONNX] {path.name}: {session.get_providers()}", flush=True)
    return session


def infer(session: ort.InferenceSession, tensor: np.ndarray) -> list:
    """按模型声明的浮点输入类型运行推理,不统计主机性能."""
    details = session.get_inputs()
    if len(details) != 1 or details[0].type not in ("tensor(float)", "tensor(float16)"):
        raise ValueError("基准模型必须只有一个 FP32 或 FP16 输入.")
    dtype = np.float16 if details[0].type == "tensor(float16)" else np.float32
    outputs = session.run(None, {details[0].name: tensor.astype(dtype)})
    if any(not np.isfinite(value).all() for value in outputs):
        raise ValueError("ONNX 输出包含非有限数值.")
    return outputs


def evaluate_vit(args, deploy: Path) -> tuple:
    """按板端 OpenCV 缩放和中心裁剪协议计算 ImageNet Top-1."""
    helper = load_module("vit_metrics", deploy / "evaluate_full_accuracy.py")
    labels = helper.read_labels(args.dataset_root / "val_labels_0based.txt")
    paths = sorted((args.dataset_root / "val").glob("*.JPEG"))
    if len(paths) != 50000:
        raise ValueError("ImageNet 验证集必须包含 50000 张图片.")
    session = make_session(args.onnx)
    correct = 0
    for index, path in enumerate(tqdm.tqdm(paths, desc="ONNX Top-1", unit="img")):
        if path.name != f"ILSVRC2012_val_{index + 1:08d}.JPEG":
            raise ValueError(f"ImageNet 图片编号不连续: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"无法读取图片: {path}")
        ratio = 256.0 / min(image.shape[:2])
        width, height = round(image.shape[1] * ratio), round(image.shape[0] * ratio)
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        top, left = (height - 224) // 2, (width - 224) // 2
        crop = resized[top:top + 224, left:left + 224, ::-1]
        tensor = crop.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        scores = infer(session, tensor)[0].reshape(-1)
        if scores.size != 1000:
            raise ValueError("分类输出必须有 1000 个类别.")
        correct += int(scores.argmax()) == labels[index]
    return "Top-1", correct / len(paths), len(paths)


def evaluate_face(args, deploy: Path) -> tuple:
    """复用对齐人脸预处理与 LFW 十折阈值协议."""
    from face_utils import load_aligned_face
    helper = load_module("face_metrics", deploy / "full_accuracy_board.py")
    pairs = helper.load_pairs(args.dataset_root)
    names = sorted({row[key] for row in pairs for key in ("image_a", "image_b")})
    session = make_session(args.onnx)
    features = {}
    for name in tqdm.tqdm(names, desc="ONNX LFW", unit="img"):
        values = infer(session, load_aligned_face(args.dataset_root / "images" / name))[0]
        values = values.astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(values))
        if values.size != 128 or not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"人脸特征无效: {name}")
        features[name] = values / norm
    metrics, _ = helper.evaluate_pairs(pairs, features)
    return "LFW 验证准确率", metrics["verification_accuracy"], len(pairs)


def evaluate_depth(args, deploy: Path) -> tuple:
    """复用 DA-2K 预处理、原图坐标恢复和点对判定."""
    from depth_utils import preprocess, INPUT_SIZE
    helper = load_module("depth_metrics", deploy / "full_accuracy_board.py")
    annotations = helper.load_protocol(args.dataset_root)
    session = make_session(args.onnx)
    correct = total = 0
    for name in tqdm.tqdm(sorted(annotations), desc="ONNX DA-2K", unit="img"):
        path = args.dataset_root / name
        depth = infer(session, preprocess(path))[0].squeeze()
        if depth.shape != (INPUT_SIZE, INPUT_SIZE) or depth.std() <= 0:
            raise ValueError(f"深度输出形状或数值无效: {name}")
        count, pairs = helper.score_pairs(path, depth, annotations[name])
        correct += count
        total += pairs
    return "DA-2K 点对准确率", correct / total, total


def coco_score(annotation, records: list, kind: str, index: int) -> float:
    """隐藏非核心 COCO 指标打印,同时正确处理没有检测结果的情况."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    print("[ONNX] 计算 COCO 核心精度,请等待.", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        if records:
            prediction = annotation.loadRes(records)
        else:
            prediction = COCO()
            prediction.dataset = {"images": annotation.dataset["images"],
                                  "categories": annotation.dataset["categories"],
                                  "annotations": []}
            prediction.createIndex()
        evaluator = COCOeval(annotation, prediction, kind)
        evaluator.params.imgIds = sorted(annotation.getImgIds())
        if kind == "segm":
            evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return float(evaluator.stats[index])


def evaluate_coco(args, deploy: Path) -> tuple:
    """评测 FastSAM 分割或 YOLO-World 检测,复用已有后处理."""
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_utils
    if args.model == "fastsam":
        from fastsam_utils import preprocess, postprocess
        helper = load_module("fastsam_metrics", deploy / "full_accuracy_board.py")
        with contextlib.redirect_stdout(io.StringIO()):
            annotation = helper.make_class_agnostic_ground_truth(args.dataset_root /
                "annotations/instances_val2017.json")
        protocol = json.loads((deploy / "accuracy_protocol.json").read_text(encoding="utf-8"))
    else:
        from yoloworld_utils import preprocess_image, decode_outputs, COCO_CLASSES
        with contextlib.redirect_stdout(io.StringIO()):
            annotation = COCO(str(args.dataset_root / "annotations/instances_val2017.json"))
        categories = {item["name"]: item["id"] for item in annotation.dataset["categories"]}
        if set(categories) != set(COCO_CLASSES):
            raise ValueError("COCO 类别不匹配.")
    paths = sorted((args.dataset_root / "images").glob("*.jpg"))
    if len(paths) != 5000 or {int(path.stem) for path in paths} != set(annotation.getImgIds()):
        raise ValueError("COCO 图片与标注必须覆盖相同的 5000 张图片.")
    session = make_session(args.onnx)
    records = []
    for path in tqdm.tqdm(paths, desc=f"ONNX {args.model}", unit="img"):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"无法读取图片: {path}")
        if args.model == "fastsam":
            tensor, geometry = preprocess(image)
            _, scores, masks = postprocess(infer(session, tensor), geometry,
                protocol["confidence"], protocol["nms_iou"], protocol["max_detections"])
            for score, mask in zip(scores, masks):
                encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                encoded["counts"] = encoded["counts"].decode("ascii")
                records.append({"image_id": int(path.stem), "category_id": 1,
                                "segmentation": encoded, "score": float(score)})
        else:
            tensor, geometry = preprocess_image(image)
            detections = decode_outputs(infer(session, tensor), geometry, 0.001, 0.65, 300)
            for item in detections:
                x1, y1, x2, y2 = item["bbox_xyxy"]
                records.append({"image_id": int(path.stem),
                    "category_id": categories[COCO_CLASSES[item["class_id"]]],
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": item["score"]})
    segmentation = args.model == "fastsam"
    value = coco_score(annotation, records, "segm" if segmentation else "bbox",
                       8 if segmentation else 0)
    return "segm AR@100" if segmentation else "mAP@0.5:0.95", value, len(paths)


def evaluate_pose(args, deploy: Path) -> tuple:
    """复用官方人体检测框、SimCC 解码和 WholeBody OKS-NMS 协议."""
    from rtmpose_utils import preprocess_image, decode_prediction
    metrics = load_module("pose_metrics", deploy / "inference_demo/evaluate_coco_wholebody.py")
    detections = json.loads((args.dataset_root /
        "person_detection_results/COCO_val2017_detections_AP_H_56_person.json").read_text())
    if len(detections) != 104125:
        raise ValueError("人体检测框必须有 104125 个.")
    ordered = sorted(enumerate(detections), key=lambda item: (item[1]["image_id"], item[0]))
    session = make_session(args.onnx)
    current_id = None
    path = args.work_dir / "pose_predictions.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for index, record in tqdm.tqdm(ordered, desc="ONNX WholeBody", unit="det"):
            if record["category_id"] != 1 or min(record["bbox"][2:]) <= 0:
                raise ValueError("无效的人体检测框.")
            if current_id != record["image_id"]:
                current_id = record["image_id"]
                image = cv2.imread(str(args.dataset_root / "images" / f"{current_id:012d}.jpg"))
                if image is None:
                    raise ValueError(f"缺少图片: {current_id}")
            tensor, geometry = preprocess_image(image, record["bbox"])
            outputs = infer(session, tensor)
            if len(outputs) != 2:
                raise ValueError("RTMPose 必须输出 X/Y 两个 SimCC 检测头.")
            # 根据宽度排列两个输出,不依赖导出顺序.
            outputs = sorted(outputs, key=lambda value: value.shape[-1])
            keypoints, area = decode_prediction(*outputs, geometry)
            stream.write(json.dumps({"detection_id": index, "image_id": current_id,
                "bbox_score": record["score"], "area": area, "keypoints": keypoints}) + "\n")
    metrics.evaluate(SimpleNamespace(predictions=path,
        annotations=args.dataset_root / "annotations/coco_wholebody_val_v1.0.json",
        formatted=args.work_dir / "formatted.json", metrics=args.work_dir / "metrics.json",
        summary_log=args.work_dir / "cocoeval.log"))
    value = json.loads((args.work_dir / "metrics.json").read_text())["metrics"]["wholebody"]["AP"]
    return "WholeBody AP", value, len(detections)


def load_sources(dataset_root: Path) -> dict[str, dict]:
    """从 test-clean 官方转录文件核对全部音频与参考文本。"""
    sources = {}
    for transcript in sorted(dataset_root.rglob("*.trans.txt")):
        for line in transcript.read_text(encoding="utf-8").splitlines():
            sample_id, reference_text = line.split(" ", 1)
            audio = transcript.parent / f"{sample_id}.flac"
            if sample_id in sources or not audio.is_file():
                raise ValueError(f"重复样例或缺少音频: {sample_id}。")
            sources[sample_id] = {"sample_id": sample_id,
                                  "reference_text": reference_text,
                                  "audio": audio}
    if len(sources) != 2620:
        raise ValueError(f"test-clean 样例不完整: {len(sources)}。")
    return sources


def evaluate_whisper(args, deploy: Path) -> tuple:
    """用双 ONNX 和板端相同的固定 Cache、英文 Greedy 规则计算 WER."""
    import whisper
    from whisper.tokenizer import get_tokenizer
    helper = load_module("whisper_metrics", deploy / "evaluate_accuracy.py")
    assets = load_module("whisper_assets", deploy / "export_board_assets.py")
    sources = load_sources(args.dataset_root)
    assets.export_assets(args.work_dir)
    config = dict(line.split("=", 1) for line in
                  (args.work_dir / "decode_config.txt").read_text().splitlines())
    initial = [int(value) for value in config["initial"].split(",")]
    suppressed = [int(value) for value in config["suppress"].split(",")]
    first = [int(value) for value in config["suppress_first"].split(",")]
    encoder, decoder = make_session(args.onnx), make_session(args.decoder_onnx)
    decoder_inputs = {item.name: item for item in decoder.get_inputs()}
    cache_names = [item.name for item in decoder.get_inputs() if "_cache_" in item.name]
    if len(cache_names) != 8:
        raise ValueError("Whisper-Tiny 必须有四层共八个固定 KV Cache 输入.")
    cache_length = decoder_inputs[cache_names[0]].shape[-1]
    tokenizer = get_tokenizer(multilingual=True, language="en", task="transcribe")
    predictions = {}
    for sample_id in tqdm.tqdm(sorted(sources), desc="ONNX WER", unit="audio"):
        decoded = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i",
            str(sources[sample_id]["audio"]), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
            check=True, stdout=subprocess.PIPE).stdout
        audio = np.frombuffer(decoded, dtype="<i2").astype(np.float32) / 32768.0
        if not audio.size:
            raise ValueError(f"空音频: {sample_id}")
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=80).numpy()[None]
        features = infer(encoder, mel)[0]
        caches = {name: np.zeros(decoder_inputs[name].shape, np.float32) for name in cache_names}
        tokens, token = [], initial[0]
        for position in range(cache_length):
            feeds = {
                "token_onehot": np.zeros(decoder_inputs["token_onehot"].shape, np.float32),
                "audio_features": features,
                "position_weights": np.zeros(decoder_inputs["position_weights"].shape, np.float32),
                "cache_update_mask": np.zeros(decoder_inputs["cache_update_mask"].shape, np.float32),
                "attention_mask": np.full(decoder_inputs["attention_mask"].shape, -10000, np.float32),
                **caches,
            }
            feeds["token_onehot"][0, token] = 1
            feeds["position_weights"][0, position] = 1
            feeds["cache_update_mask"][..., position] = 1
            feeds["attention_mask"][..., :position + 1] = 0
            for name, value in feeds.items():
                dtype = np.float16 if decoder_inputs[name].type == "tensor(float16)" else np.float32
                feeds[name] = value.astype(dtype)
            outputs = decoder.run(None, feeds)
            if any(not np.isfinite(value).all() for value in outputs):
                raise ValueError(f"Decoder 输出无效: {sample_id}")
            output_map = dict(zip([item.name for item in decoder.get_outputs()], outputs))
            caches = {name: output_map[name.removesuffix("_in") + "_out"] for name in cache_names}
            if position + 1 < len(initial):
                token = initial[position + 1]
                continue
            logits = outputs[0].astype(np.float32).reshape(-1)
            logits[suppressed] = -np.inf
            if not tokens:
                logits[first] = -np.inf
            token = int(logits.argmax())
            if token == int(config["eot"]):
                break
            tokens.append(token)
            if len(tokens) >= int(config["max_generated_tokens"]):
                break
        predictions[sample_id] = {"sample_id": sample_id, "tokens": tokens,
                                 "text": tokenizer.decode(tokens).strip()}
    accuracy, _ = helper.metric_summary("librispeech", sources, predictions)
    return "WER", accuracy["value"], len(predictions)


def main() -> None:
    """只保存核心精度,成功后清理本次专用临时目录."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--decoder-onnx", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.model == "whisper_tiny" and args.decoder_onnx is None:
        parser.error("Whisper 精度评测需要 --decoder-onnx.")
    deploy = ROOT / "models" / MODELS[args.model] / "deploy"
    sys.path.insert(0, str(deploy))
    sys.path.insert(0, str(deploy / "inference_demo"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 临时目录由工具创建,成功后仅清理此目录; 失败保留现场.
    import tempfile
    args.work_dir = Path(tempfile.mkdtemp(prefix="onnx_accuracy_", dir=args.output.parent))
    adapters = {"vit_base_patch16_224": evaluate_vit, "mobilefacenet": evaluate_face,
        "depth_anything_v2_small": evaluate_depth, "rtmpose_body2d": evaluate_pose,
        "fastsam": evaluate_coco, "yoloworld_xl": evaluate_coco, "whisper_tiny": evaluate_whisper}
    metric, value, samples = adapters[args.model](args, deploy)
    value = float(value)
    if not math.isfinite(value) or value < 0 or (metric != "WER" and value > 1):
        raise ValueError("ONNX 核心精度无效.")
    args.output.write_text(json.dumps({"status": "complete", "model": args.model,
        "backend": "ONNX Runtime", "metric": metric, "accuracy": value,
        "samples": samples, "source": "本次 ONNX 浮点全量实测,使用与板端相同的评测协议"},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(args.work_dir)
    print(f"[ONNX] {metric}: {value:.6f}; 汇总: {args.output}", flush=True)


if __name__ == "__main__":
    main()
