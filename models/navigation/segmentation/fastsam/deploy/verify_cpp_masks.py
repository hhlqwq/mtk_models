"""用同一次 NPU 原始输出核对 C++ 与参考后处理的实例掩码."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fastsam_utils import OUTPUT_NAMES, postprocess, sha256_file


def main():
    """逐实例比较板端 PNG 与 NumPy 复算结果,保存完整匹配清单."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--board-result", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.board_result / "results.json").read_text(encoding="utf-8"))
    geometry = json.loads(args.export_manifest.read_text(encoding="utf-8"))["geometry"]
    with np.load(args.raw, allow_pickle=False) as archive:
        outputs = [archive[name] for name in OUTPUT_NAMES]
    boxes, scores, masks = postprocess(outputs, geometry, max_det=100)
    if len(masks) != run["instance_count_before_prompt"]:
        raise ValueError("C++ 与参考后处理的实例数量不一致.")
    items = []
    for record in run["detections"]:
        index = record["index"]
        path = args.board_result / record["mask"]
        png = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if png is None or png.shape != masks[index].shape:
            raise ValueError(f"掩码丢失或尺寸错误: {path}")
        board = png > 0
        expected = masks[index]
        intersection = np.logical_and(board, expected).sum()
        union = np.logical_or(board, expected).sum()
        overlap = float(intersection / union) if union else 1.0
        items.append({"index": index, "mask_iou": overlap,
                      "board_pixels": int(board.sum()),
                      "reference_pixels": int(expected.sum()),
                      "mask_sha256": sha256_file(path),
                      "box_max_absolute_error": float(np.max(
                          np.abs(boxes[index] - record["box_xyxy"]))),
                      "box_delta": (boxes[index] - record["box_xyxy"]).tolist(),
                      "score_absolute_error": float(abs(scores[index] - record["score"]))})
    report = {
        "protocol": "same_npu_raw_output_cpp_vs_numpy_postprocessing",
        "npu_archive_sha256": sha256_file(args.raw),
        "results_sha256": sha256_file(args.board_result / "results.json"),
        "compared_instances": len(items),
        "mean_mask_iou": float(np.mean([item["mask_iou"] for item in items])),
        "min_mask_iou": float(min(item["mask_iou"] for item in items)),
        "max_box_absolute_error": max(item["box_max_absolute_error"] for item in items),
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"[VERIFY] C++/NumPy 掩码 IoU 平均 {report['mean_mask_iou']:.6f},"
          f" 最低 {report['min_mask_iou']:.6f}.")


if __name__ == "__main__":
    main()
