"""将各模型板端工作目录汇总为一个核心指标文件."""

import argparse
import json
import math
import shutil
from pathlib import Path

from tools.files import read_json

import os
from tools.evaluation import (
    add_test_argument, is_test, matching_reference, result_status, scope_metadata,
)

from tools.board import configure_parser


def require_count(report: dict, key: str, count: int) -> None:
    """验证全量样本计数,避免不完整结果被标为成功."""
    if report.get(key) != count:
        raise ValueError(f"{key} 不完整: {report.get(key)}/{count}.")


def extract_result(model: str, work_dir: Path) -> dict:
    """从各模型的实际字段读取核心精度和独立的 NPU 耗时."""
    if model == "rtmpose_body2d":
        report = read_json(work_dir / "coco_wholebody_metrics.json")
        count = report["protocol"]["detections_before_nms"] if is_test() else 104125
        if is_test():
            require_count(report["protocol"], "evaluated_images", 100)
        require_count(report["protocol"], "detections_before_nms", count)
        if report.get("sample_ids_sha256"):
            os.environ["MTK_EVAL_SAMPLE_SHA256"] = report["sample_ids_sha256"]
        timing = read_json(work_dir / "timing_summary.json")
        require_count(timing, "processed_detections", count)
        return {
            "dataset": "coco_wholebody_val2017", "samples": count,
            "images": report["protocol"]["evaluated_images"],
            "metric": "WholeBody AP", "board_accuracy": report["metrics"]["wholebody"]["AP"],
            "npu_mean_ms": timing["npu_mean_ms"],
            "peak_rss_kib": timing.get("peak_rss_kb"),
            "timing_scope": "常驻模型,每个人体裁剪的 NeuronRuntime_inference 调用",
        }
    report = read_json(work_dir / "report" / "summary.json")
    if report.get("status") != result_status():
        raise ValueError("板端报告未完成.")
    if is_test() and (report.get("evaluation_scope") != "subset" or not report.get("sample_ids_sha256")):
        raise ValueError("快速报告缺少范围或样本指纹.")
    if report.get("sample_ids_sha256"):
        os.environ["MTK_EVAL_SAMPLE_SHA256"] = report["sample_ids_sha256"]
    result = {"dataset": report["dataset"], "peak_rss_kib": report.get("peak_rss_kib")}
    if model == "vit_base_patch16_224":
        require_count(report, "samples", 200 if is_test() else 50000)
        result.update(samples=200 if is_test() else 50000, metric="Top-1", board_accuracy=report["npu_top1"],
                      npu_mean_ms=report["npu_mean_ms"],
                      timing_scope="常驻模型,所选图片的 NeuronRuntime_inference 调用")
    elif model in ("mobilefacenet", "depth_anything_v2_small"):
        if model == "mobilefacenet":
            require_count(report, "pairs", 100 if is_test() else 6000)
            metric, key, count = "LFW 验证准确率", "verification_accuracy", 100 if is_test() else 6000
        else:
            require_count(report, "images", 100 if is_test() else 1033)
            if is_test():
                if type(report.get("total_pairs")) is not int or report["total_pairs"] < 100:
                    raise ValueError("DA-2K 快速报告的点对计数无效.")
            else:
                require_count(report, "total_pairs", 2068)
            metric, key, count = "DA-2K 点对准确率", "pairwise_accuracy", report["total_pairs"] if is_test() else 2068
        timing = read_json(work_dir / "report" / "benchmark.json")
        result.update(samples=count, metric=metric, board_accuracy=report[key],
                      npu_mean_ms=timing["mean_ms"],
                      peak_rss_kib=timing.get("peak_rss_kib"),
                      timing_scope="独立 C++ 常驻模型,按本次模式预热并重复同一输入")
    elif model == "fastsam":
        require_count(report, "images", 100 if is_test() else 5000)
        result.update(samples=100 if is_test() else 5000, metric="segm AR@100", board_accuracy=report["AR_100"],
                      npu_mean_ms=report["npu_mean_ms"],
                      timing_scope="逐图加载模型,单独统计 NeuronRuntime_inference 调用")
    elif model == "whisper_tiny":
        require_count(report, "expected_samples", 5 if is_test() else 2620)
        require_count(report, "successful_samples", 5 if is_test() else 2620)
        result["peak_rss_kib"] = report["performance"]["overall"]["peak_rss_kb"].get("max")
        result.update(samples=5 if is_test() else 2620, metric="WER", board_accuracy=report["accuracy"]["value"],
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
    # Linux ru_maxrss 的单位是 KiB,统一换算为 MiB.
    peak_rss = result.pop("peak_rss_kib", None)
    if peak_rss is not None and (not math.isfinite(float(peak_rss)) or float(peak_rss) <= 0):
        raise ValueError("峰值 RSS 无效.")
    result["peak_rss_mib"] = None if peak_rss is None else float(peak_rss) / 1024.0
    result["memory_scope"] = "推理进程峰值 RSS"
    if args.model in ("mobilefacenet", "depth_anything_v2_small"):
        result["memory_scope"] = "独立 C++ 常驻推理进程峰值 RSS"
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
    if args.reference_report and args.reference_report.is_file():
        report = read_json(args.reference_report)
        if matching_reference(report):
            reference = report["accuracy"]
            args.reference_source = report["source"]
    if is_test():
        reference_report = read_json(args.reference_report) if args.reference_report and args.reference_report.is_file() else {}
        if not matching_reference(reference_report):
            reference = None
    if reference is not None and (not math.isfinite(reference) or reference < 0 or (
            result["metric"] != "WER" and reference > 1)):
        raise ValueError("参考精度无效,准确率需填写 0 到 1 之间的数值.")
    # 错误率越低越好,其余准确率越高越好; 正差值统一表示精度下降.
    loss = None if reference is None else (
        accuracy - reference if result["metric"] == "WER" else reference - accuracy) * 100
    summary = {
        "status": result_status(), **scope_metadata(), "model": args.model, "run_id": args.run_id,
        **result, "reference_accuracy": reference,
        "accuracy_loss_percentage_points": loss,
        "accuracy_change_percentage_points": None if loss is None else -loss,
        "reference_source": args.reference_source if reference is not None else None,
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    if result["npu_mean_ms"] is None:
        print(f"板端 Runtime 平均耗时: {result['runtime_mean_ms']:.3f} ms; 纯 NPU 耗时未测量.")
    else:
        print(f"板端 NPU 平均推理耗时: {result['npu_mean_ms']:.3f} ms")
    if result["peak_rss_mib"] is None:
        print("推理进程峰值 RSS: 未记录.")
    else:
        print(f"推理进程峰值 RSS: {result['peak_rss_mib']:.3f} MiB")
    print(f"板端 {result['metric']}: {accuracy:.6f}")
    if reference is None:
        print("精度下降: 未计算,缺少匹配的参考基准.")
    else:
        print(f"参考 {result['metric']}: {reference:.6f}")
        print(f"精度变化: {-loss:+.4f} 个百分点; 正数为改善,负数为下降; 来源: {args.reference_source}")
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
    parser.add_argument("--reference-report", type=Path, help="参考报告,快速模式须匹配子集指纹.")
    add_test_argument(parser)
    configure_parser(parser, __file__, {
        "model": lambda c: c.config["name"],
        "work_dir": lambda c: c.work,
        "output": lambda c: c.result / "summary.json",
        "run_id": lambda c: c.run_id,
        "reference_report": lambda c: c.root / "models" / ("onnx_reference_test.json" if is_test() else "onnx_reference.json"),
    })
    return parser.parse_args()


if __name__ == "__main__":
    summarize(parse_args())
