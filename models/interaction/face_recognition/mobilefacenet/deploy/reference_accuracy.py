"""按板端相同的非对齐 LFW 十折协议评测 FP32 ONNX。"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from face_utils import load_aligned_face
from full_accuracy_board import evaluate_pairs, load_pairs, sha256_file


def main() -> None:
    """核对全量输入，逐图提取浮点特征并保存十折结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_pairs(args.dataset_root)
    names = sorted({row[key] for row in rows for key in ("image_a", "image_b")})
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    session = ort.InferenceSession(
        str(args.model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("FP32 ONNX CUDA EP 不可用。")
    input_name = session.get_inputs()[0].name
    features = {}
    for index, name in enumerate(names, 1):
        values = session.run(None, {input_name: load_aligned_face(
            args.dataset_root / "images" / name)})[0].reshape(-1)
        norm = float(np.linalg.norm(values))
        if values.size != 128 or not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"FP32 特征无效: {name}。")
        features[name] = values.astype(np.float32) / norm
        if index % 100 == 0 or index == len(names):
            print(f"[PROGRESS] LFW FP32 {index}/{len(names)} 张。", flush=True)
    verification, details = evaluate_pairs(rows, features)
    report = {
        "status": "complete",
        "backend": "onnxruntime_cuda_fp32",
        "dataset": "lfw_original_non_aligned_6000_pairs_10_folds",
        "unique_images": len(names),
        "model_sha256": sha256_file(args.model),
        "pairs_sha256": sha256_file(args.dataset_root / "pairs.csv"),
        **verification,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "pair_results.jsonl").open("w", encoding="utf-8") as stream:
        for item in details:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[RESULT] LFW FP32 准确率 {report['verification_accuracy']:.6f}。", flush=True)


if __name__ == "__main__":
    main()
