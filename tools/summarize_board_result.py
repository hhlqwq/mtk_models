"""将各模型板端工作目录汇总为一个核心指标文件."""

import argparse
import json
import math
import shutil
from pathlib import Path


def read_json(path: Path) -> dict:
    """读取模型已有的机器可读报告."""
    return json.loads(path.read_text(encoding="utf-8"))


def require_count(report: dict, key: str, count: int) -> None:
    """验证全量样本计数,避免不完整结果被标为成功."""
    if report.get(key) != count:
        raise ValueError(f"{key} 不完整: {report.get(key)}/{count}.")


def extract_result(model: str, work_dir: Path) -> dict:
    """从各模型的实际字段读取核心精度和独立的 NPU 耗时."""
    if model == "rtmpose_body2d":
        report = read_json(work_dir / "coco_wholebody_metrics.json")
        require_count(report["protocol"], "detections_before_nms", 104125)
        timing = read_json(work_dir / "timing_summary.json")
        require_count(timing, "processed_detections", 104125)
        return {
            "dataset": "coco_wholebody_val2017", "samples": 104125,
            "metric": "WholeBody AP", "board_accuracy": report["metrics"]["wholebody"]["AP"],
            "npu_mean_ms": timing["npu_mean_ms"],
            "timing_scope": "常驻模型,每个人体裁剪的 NeuronRuntime_inference 调用",
        }
    report = read_json(work_dir / "report" / "summary.json")
    if report.get("status") != "complete":
        raise ValueError("板端报告未完成.")
    result = {"dataset": report["dataset"]}
    if model == "vit_base_patch16_224":
        require_count(report, "samples", 50000)
        result.update(samples=50000, metric="Top-1", board_accuracy=report["npu_top1"],
                      npu_mean_ms=report["npu_mean_ms"],
                      timing_scope="常驻模型,50000 张图片的 NeuronRuntime_inference 调用")
    elif model in ("mobilefacenet", "depth_anything_v2_small"):
        if model == "mobilefacenet":
            require_count(report, "pairs", 6000)
            metric, key, count = "LFW 验证准确率", "verification_accuracy", 6000
        else:
            require_count(report, "images", 1033)
            require_count(report, "total_pairs", 2068)
            metric, key, count = "DA-2K 点对准确率", "pairwise_accuracy", 2068
        timing = read_json(work_dir / "report" / "benchmark.json")
        result.update(samples=count, metric=metric, board_accuracy=report[key],
                      npu_mean_ms=timing["mean_ms"],
                      timing_scope="独立 C++ 常驻模型,预热 10 次后重复同一输入 100 次")
    elif model == "fastsam":
        require_count(report, "images", 5000)
        result.update(samples=5000, metric="segm AR@100", board_accuracy=report["AR_100"],
                      npu_mean_ms=report["npu_mean_ms"],
                      timing_scope="逐图加载模型,单独统计 NeuronRuntime_inference 调用")
    elif model == "yoloworld_xl":
        require_count(report, "images", 5000)
        result.update(samples=5000, metric="mAP@0.5:0.95",
                      board_accuracy=report["accuracy"]["AP_50_95"], npu_mean_ms=None,
                      runtime_mean_ms=report["timing"]["mean_ms"],
                      timing_scope="ONNX Runtime session.Run,非独立 NPU 计时",
                      provider_node_events=report["profile"]["provider_node_events"])
    elif model == "whisper_tiny":
        require_count(report, "expected_samples", 2620)
        require_count(report, "successful_samples", 2620)
        result.update(samples=2620, metric="WER", board_accuracy=report["accuracy"]["value"],
                      npu_mean_ms=report["performance"]["overall"]["npu_total_ms"]["mean"],
                      timing_scope="每条音频的 Encoder 与所有 Decoder NPU 调用之和")
    else:
        raise ValueError(f"不支持的模型: {model}.")
    return result


def summarize(args: argparse.Namespace) -> None:
    """写入精简报告,成功后仅删除当前运行的专用 work 目录."""
    work_dir = args.work_dir.resolve()
    output = args.output.resolve()
    if (args.work_dir.is_symlink() or work_dir.name != "work" or
            output.parent != work_dir.parent or output.name != "summary.json"):
        raise ValueError("汇总必须位于本次 work 目录的同级 summary.json.")
    result = extract_result(args.model, work_dir)
    accuracy = float(result["board_accuracy"])
    if not math.isfinite(accuracy) or accuracy < 0 or (
            result["metric"] != "WER" and accuracy > 1):
        raise ValueError("核心精度指标无效.")
    for key in ("npu_mean_ms", "runtime_mean_ms"):
        if result.get(key) is not None:
            value = float(result[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} 无效.")
    reference = float(args.reference) if args.reference else None
    if reference is not None and (not math.isfinite(reference) or reference < 0 or (
            result["metric"] != "WER" and reference > 1)):
        raise ValueError("参考精度无效,准确率需填写 0 到 1 之间的数值.")
    # 错误率越低越好,其余准确率越高越好; 正差值统一表示精度下降.
    loss = None if reference is None else (
        accuracy - reference if result["metric"] == "WER" else reference - accuracy) * 100
    summary = {
        "status": "complete", "model": args.model, "run_id": args.run_id,
        **result, "reference_accuracy": reference,
        "accuracy_loss_percentage_points": loss,
        "reference_source": args.reference_source if reference is not None else None,
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    if result["npu_mean_ms"] is None:
        print(f"板端 Runtime 平均耗时: {result['runtime_mean_ms']:.3f} ms; 纯 NPU 耗时未测量.")
    else:
        print(f"板端 NPU 平均推理耗时: {result['npu_mean_ms']:.3f} ms")
    print(f"板端 {result['metric']}: {accuracy:.6f}")
    if reference is None:
        print("精度下降: 未计算,缺少匹配的参考基准.")
    else:
        print(f"参考 {result['metric']}: {reference:.6f}")
        print(f"精度下降: {loss:.4f} 个百分点; 来源: {args.reference_source}")
    shutil.rmtree(work_dir)
    print(f"[OK] 结果: {output}")


def parse_args() -> argparse.Namespace:
    """解析模型、工作目录和同协议参考精度."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reference", default="")
    parser.add_argument("--reference-source", default="用户提供的同协议参考基准")
    return parser.parse_args()


if __name__ == "__main__":
    summarize(parse_args())
