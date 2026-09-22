"""在 Genio 720 执行 FastSAM-s NPU 或 ONNX 推理,保存原始输出与可视结果."""

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastsam_utils import (OUTPUT_NAMES, postprocess, preprocess, select_masks,
                           sha256_file)


def load_native(path, detail):
    """按照 NCHW INT8 与 16 元素行对齐契约读取 MDLA 原生输出."""
    values = np.fromfile(path, dtype=np.int8)
    n, channels, height, width = detail["shape"]
    padded_width = (width + 15) // 16 * 16
    if values.size == n * channels * height * width:
        values = values.reshape(n, channels, height, width)
    elif values.size == n * channels * height * padded_width:
        values = values.reshape(n, channels, height, padded_width)[..., :width]
    else:
        raise ValueError(f"原生输出大小不匹配: {path}, {values.size}.")
    return (values.astype(np.float32) - detail["zero_point"]) * detail["scale"]


def run_npu(args, tensor):
    """调用 neuronrt 硬件模式并返回反量化输出,保存完整运行日志."""
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata.get("native_layout") != "NCHW_INT8_WIDTH_ALIGN16":
        raise ValueError("尚未支持该 MDLA 输出布局.")
    detail = metadata["input"]
    if detail["dtype"] != "int8" or detail["shape"] != list(tensor.shape):
        raise ValueError("输入张量契约不匹配.")
    quantized = np.clip(np.rint(tensor / detail["scale"]) + detail["zero_point"],
                        -128, 127).astype(np.int8)
    input_path = args.output_dir / "input_int8.bin"
    quantized.tofile(input_path)
    command = [args.neuronrt, "-m", "hw", "-a", str(args.model),
               "-i", str(input_path), "-b", "100", "-r", "turbo", "-l", "performance"]
    for item in metadata["outputs"]:
        command.extend(["-o", str(args.output_dir / f"output_{item['index']}.bin")])
    with (args.output_dir / "neuronrt.log").open("w", encoding="utf-8") as log:
        subprocess.run(command + ["--verbose"], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=args.timeout)
    outputs = {item["semantic"]: load_native(
        args.output_dir / f"output_{item['index']}.bin", item)
               for item in metadata["outputs"]}
    return [outputs[name] for name in OUTPUT_NAMES], command


def benchmark(args, command):
    """单独记录预热和计时日志,不把进程启动时间标成纯 NPU 延迟."""
    for label, count in (("warmup", 10), ("benchmark", args.iterations)):
        print(f"[BENCHMARK] {label}: {count}", flush=True)
        with (args.output_dir / f"{label}.log").open("w", encoding="utf-8") as log:
            # 性能测量不覆盖用于一致性比较的单次输出.
            performance_command = []
            index = 0
            while index < len(command):
                if command[index] == "-o":
                    index += 2
                else:
                    performance_command.append(command[index])
                    index += 1
            subprocess.run(performance_command + ["-c", str(count), "--verbose"],
                           stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=args.timeout)


def main():
    """执行单图推理,保存掩码、原始张量、资源记录与端到端耗时."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("npu", "onnx"), default="npu")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neuronrt", default="/usr/sbin/neuronrt")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.9)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--point", type=int, nargs=2)
    group.add_argument("--box", type=int, nargs=4)
    args = parser.parse_args()
    if args.backend == "npu" and args.metadata is None:
        parser.error("NPU 后端需要 --metadata.")
    if args.iterations < 0 or args.timeout <= 0:
        parser.error("iterations 不能为负, timeout 必须为正.")
    manifest_path = args.model.parent / "deployment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected_hash in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("部署清单只允许模型目录内文件.")
        if sha256_file(args.model.parent / name) != expected_hash:
            raise ValueError(f"部署文件哈希不匹配: {name}.")
    if args.model.name not in manifest["files"]:
        raise ValueError("模型未列入部署清单.")
    if args.metadata is not None and (
            args.metadata.resolve() != (args.model.parent / "model_int8.json").resolve()):
        raise ValueError("必须使用同一部署清单中的量化元数据.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(f"[1/4] 准备图片,后端 {args.backend}.", flush=True)
    start = time.perf_counter()
    image = cv2.imread(str(args.image))
    tensor, geometry = preprocess(image)
    infer_start = time.perf_counter()
    command = None
    if args.backend == "npu":
        outputs, command = run_npu(args, tensor)
    else:
        import onnxruntime as ort
        session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
        outputs = session.run(OUTPUT_NAMES, {"images": tensor})
    infer_ms = (time.perf_counter() - infer_start) * 1000
    print("[2/4] 解码实例掩码.", flush=True)
    boxes, scores, masks = postprocess(outputs, geometry, args.confidence,
                                       args.iou, args.max_det)
    selected = select_masks(masks, args.point, args.box)
    pipeline_ms = (time.perf_counter() - start) * 1000
    print("[3/4] 保存结果与哈希.", flush=True)
    np.savez(args.output_dir / "raw_outputs.npz", images=tensor,
             **dict(zip(OUTPUT_NAMES, outputs)))
    overlay = image.copy()
    mask_directory = args.output_dir / "masks"
    mask_directory.mkdir()
    records = []
    for index in selected:
        color = np.array([(index * 67 + 43) % 256, (index * 97 + 101) % 256,
                          (index * 137 + 179) % 256], dtype=np.float32)
        overlay[masks[index]] = (overlay[masks[index]] * 0.5 + color * 0.5).astype(np.uint8)
        mask_path = mask_directory / f"{index:04d}.png"
        if not cv2.imwrite(str(mask_path), masks[index].astype(np.uint8) * 255):
            raise IOError(f"写入失败: {mask_path}")
        records.append({"index": index, "score": float(scores[index]),
                        "box_xyxy": boxes[index].tolist(),
                        "mask": str(mask_path.relative_to(args.output_dir)),
                        "mask_sha256": sha256_file(mask_path)})
    if not cv2.imwrite(str(args.output_dir / "overlay.jpg"), overlay):
        raise IOError("无法保存叠加图.")
    evidence = {
        "backend": args.backend, "model_sha256": sha256_file(args.model),
        "deployment_manifest_sha256": sha256_file(manifest_path),
        "image_sha256": sha256_file(args.image), "geometry": geometry,
        "confidence": args.confidence, "iou": args.iou, "max_det": args.max_det,
        "point": args.point, "box": args.box, "detections": records,
        "instance_count_before_prompt": len(masks), "command": command,
        "inference_wall_ms_including_load": infer_ms,
        "pipeline_wall_ms_excluding_output_write": pipeline_ms,
        "python_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "child_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "system": subprocess.check_output(["uname", "-a"], text=True).strip(),
        "pure_npu_latency_ms": None,
        "native_layout_verified": False if args.backend == "npu" else None,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.iterations and command:
        benchmark(args, command)
    print(f"[4/4] 完成,实例数 {len(records)}. 纯 NPU 延迟请查看 runtime 性能日志.")


if __name__ == "__main__":
    main()
