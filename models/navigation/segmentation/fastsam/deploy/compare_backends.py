"""比较同一图片的 PyTorch、ONNX 或 NPU 原始输出与匹配实例掩码."""

import argparse
import json
from pathlib import Path

import numpy as np

from fastsam_utils import OUTPUT_NAMES, box_iou, postprocess, sha256_file


def main():
    """输出张量误差与一对一匹配掩码 IoU,不冒充数据集精度."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.9)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--strict-fp32", action="store_true")
    args = parser.parse_args()
    reference = np.load(args.reference, allow_pickle=False)
    candidate = np.load(args.candidate, allow_pickle=False)
    np.testing.assert_array_equal(reference["images"], candidate["images"])
    manifest = json.loads(args.export_manifest.read_text(encoding="utf-8"))
    tensors, allclose = {}, True
    for name in OUTPUT_NAMES:
        expected, actual = reference[name], candidate[name]
        if expected.shape != actual.shape or not np.isfinite(actual).all():
            raise ValueError(f"候选输出不合法: {name}.")
        error = np.abs(actual - expected)
        passed = bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-3))
        allclose &= passed
        tensors[name] = {"max_absolute_error": float(error.max()),
                         "mean_absolute_error": float(error.mean()),
                         "fp32_allclose": passed}
    decoded = [postprocess([archive[name] for name in OUTPUT_NAMES],
                           manifest["geometry"], args.confidence, args.iou,
                           args.max_det) for archive in (reference, candidate)]
    reference_boxes, _, reference_masks = decoded[0]
    candidate_boxes, _, candidate_masks = decoded[1]
    available = set(range(len(candidate_boxes)))
    matches = []
    for index, box in enumerate(reference_boxes):
        if not available:
            break
        indices = sorted(available)
        overlaps = box_iou(box, candidate_boxes[indices])
        best = int(np.argmax(overlaps))
        if overlaps[best] < 0.5:
            continue
        selected = indices[best]
        available.remove(selected)
        union = np.logical_or(reference_masks[index], candidate_masks[selected]).sum()
        intersection = np.logical_and(reference_masks[index], candidate_masks[selected]).sum()
        matches.append({"reference": index, "candidate": selected,
                        "box_iou": float(overlaps[best]),
                        "mask_iou": float(intersection / union) if union else 1.0})
    report = {
        "protocol": "single_image_class_agnostic_consistency_not_dataset_accuracy",
        "reference_sha256": sha256_file(args.reference),
        "candidate_sha256": sha256_file(args.candidate),
        "image_sha256": manifest["image_sha256"], "tensors": tensors,
        "fp32_allclose": allclose, "reference_instances": len(reference_masks),
        "candidate_instances": len(candidate_masks), "matched_instances": len(matches),
        "unmatched_reference": len(reference_masks) - len(matches),
        "unmatched_candidate": len(candidate_masks) - len(matches),
        "matched_mask_iou_mean": (float(np.mean([item["mask_iou"] for item in matches]))
                                  if matches else None),
        "confidence": args.confidence, "iou": args.iou, "max_det": args.max_det,
        "matches": matches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[COMPARE] 匹配 {len(matches)}, FP32 allclose={allclose}.")
    if args.strict_fp32 and not allclose:
        raise SystemExit("FP32 数值一致性未通过.")


if __name__ == "__main__":
    main()
