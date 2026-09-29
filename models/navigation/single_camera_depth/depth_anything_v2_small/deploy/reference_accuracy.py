"""按板端固定方形输入和 DA-2K 点对规则评测 FP32 ONNX。"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from depth_utils import preprocess, sha256_file
from full_accuracy_board import load_protocol, score_pairs


def main() -> None:
    """处理 DA-2K 全部图片和点对，保存可核对的浮点基线。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    annotations = load_protocol(args.dataset_root)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    session = ort.InferenceSession(
        str(args.model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("FP32 ONNX CUDA EP 不可用。")
    input_name = session.get_inputs()[0].name
    correct_total = pair_total = 0
    with (args.output_dir / "image_results.jsonl").open("w", encoding="utf-8") as stream:
        for index, (relative, pairs) in enumerate(sorted(annotations.items()), 1):
            image = args.dataset_root / relative
            depth = session.run(None, {input_name: preprocess(image)})[0].squeeze()
            if depth.shape != (518, 518) or not np.isfinite(depth).all():
                raise ValueError(f"FP32 深度输出无效: {relative}。")
            correct, total = score_pairs(image, depth, pairs)
            correct_total += correct
            pair_total += total
            stream.write(json.dumps({"image": relative, "correct_pairs": correct,
                                     "pairs": total}, ensure_ascii=False) + "\n")
            if index % 50 == 0 or index == len(annotations):
                print(f"[PROGRESS] DA-2K FP32 {index}/{len(annotations)} 张。",
                      flush=True)
    report = {
        "status": "complete",
        "backend": "onnxruntime_cuda_fp32",
        "dataset": "da2k_fixed_square_518_pair_order",
        "images": len(annotations),
        "pairs": pair_total,
        "correct_pairs": correct_total,
        "pair_accuracy": correct_total / pair_total,
        "model_sha256": sha256_file(args.model),
        "annotations_sha256": sha256_file(args.dataset_root / "annotations.json"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[RESULT] DA-2K FP32 准确率 {report['pair_accuracy']:.6f}。", flush=True)


if __name__ == "__main__":
    main()
