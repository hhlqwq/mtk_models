"""仅从板端 C++ 预测计算 ImageNet val 全量 Top-1/Top-5 指标."""

import argparse
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析指标阶段输入参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def read_labels(path: Path) -> list[int]:
    """核对官方图片顺序对应的 50000 个零基类别."""
    labels = []
    for line_number, line in enumerate(path.read_text(
            encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) not in (1, 2):
            raise ValueError(f"标签格式错误: {line_number}.")
        if len(parts) == 2 and int(parts[0]) != line_number:
            raise ValueError(f"标签序号不连续: {line_number}.")
        label = int(parts[-1])
        if not 0 <= label < 1000:
            raise ValueError(f"标签类别越界: {line_number}.")
        labels.append(label)
    if len(labels) != 50000:
        raise ValueError(f"预期 50000 个标签,实际 {len(labels)}.")
    return labels


def evaluate(args: argparse.Namespace) -> None:
    """校验全部 C++ 输出并汇总精度与纯 NPU 耗时."""
    labels = read_labels(args.labels)
    top1 = top5 = holdout_top1 = holdout_top5 = 0
    timings = []
    with args.predictions.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= 50000:
                raise ValueError("预测数量超过 50000.")
            record = json.loads(line)
            expected_image = f"ILSVRC2012_val_{index + 1:08d}.JPEG"
            predictions = record.get("top5")
            if (record.get("index") != index or
                    record.get("image") != expected_image or
                    not isinstance(predictions, list) or
                    len(predictions) != 5 or
                    len(set(predictions)) != 5 or
                    any(not isinstance(value, int) or
                        not 0 <= value < 1000 for value in predictions)):
                raise ValueError(f"C++ 预测记录不匹配: {index}.")
            timing = float(record["npu_ms"])
            if not math.isfinite(timing) or timing < 0:
                raise ValueError(f"NPU 耗时非法: {index}.")
            timings.append(timing)
            top1 += predictions[0] == labels[index]
            top5 += labels[index] in predictions
            if not 1000 <= index < 1100:
                holdout_top1 += predictions[0] == labels[index]
                holdout_top5 += labels[index] in predictions
            if (index + 1) % 5000 == 0:
                print(f"[METRIC] ViT {index + 1}/50000 条.", flush=True)
    if len(timings) != 50000:
        raise ValueError(f"预测数量不完整: {len(timings)}/50000.")
    sorted_times = sorted(timings)
    summary = {
        "status": "complete",
        "model": "vit_base_patch16_224",
        "run_id": args.run_id,
        "dataset": "imagenet_val2012",
        "samples": 50000,
        "holdout_samples": 49900,
        "npu_top1": top1 / 50000,
        "npu_top5": top5 / 50000,
        "holdout_top1": holdout_top1 / 49900,
        "holdout_top5": holdout_top5 / 49900,
        "timing_scope": "persistent C++ Neuron Runtime inference call only",
        "npu_mean_ms": statistics.fmean(timings),
        "npu_p95_ms": sorted_times[math.ceil(0.95 * len(sorted_times)) - 1],
    }
    args.report.mkdir(parents=True, exist_ok=True)
    (args.report / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[OK] ViT 全量指标: {args.report}.")


if __name__ == "__main__":
    evaluate(parse_args())
