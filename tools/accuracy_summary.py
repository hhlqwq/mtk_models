"""独立全量评价只汇总精度,保留本次预测和原始报告."""

import argparse
import json
import math
from pathlib import Path

from tools.board import paths
from tools.evaluation import is_test
from tools.files import read_json


def require_count(actual, expected):
    """拒绝不完整覆盖,不把固定子集当作全量评价."""
    if actual != expected:
        raise ValueError(f"全量覆盖不完整: {actual}/{expected}.")


def vision_accuracy(ctx):
    """读取任务原始精度报告,不依赖性能验收或部署汇总."""
    model = ctx.config["name"]
    if model in ("yolov5s", "yolov8n", "yamnet"):
        report_path = ctx.result / "summary.json"
    elif model == "rtmpose_body2d":
        report_path = ctx.work / "coco_wholebody_metrics.json"
    else:
        report_path = ctx.work / "report/summary.json"
    report = read_json(report_path)
    if report.get("evaluation_scope") != "full" or report.get("status") not in (
            "complete", "board_verified"):
        raise ValueError(f"原始报告未完成全量评价: {report_path}.")
    if model == "rtmpose_body2d":
        samples = report["protocol"]["detections_before_nms"]
        require_count(samples, 104125)
        metric, value = "WholeBody AP", report["metrics"]["wholebody"]["AP"]
    elif model in ("yolov5s", "yolov8n", "yamnet"):
        samples = report["samples"]
        require_count(samples, 2000 if model == "yamnet" else 5000)
        metric, value = report["metric"], report["board_accuracy"]
    elif model == "vit_base_patch16_224":
        samples = report["samples"]
        require_count(samples, 50000)
        metric, value = "Top-1", report["npu_top1"]
    elif model == "mobilefacenet":
        samples = report["pairs"]
        require_count(samples, 6000)
        metric, value = "LFW 验证准确率", report["verification_accuracy"]
    elif model == "depth_anything_v2_small":
        require_count(report["images"], 1033)
        samples = report["total_pairs"]
        require_count(samples, 2068)
        metric, value = "DA-2K 点对准确率", report["pairwise_accuracy"]
    elif model == "fastsam":
        samples = report["images"]
        require_count(samples, 5000)
        metric, value = "segm AR@100", report["AR_100"]
    elif model == "whisper_tiny":
        samples = report["successful_samples"]
        require_count(report["expected_samples"], 2620)
        require_count(samples, 2620)
        metric, value = "WER", report["accuracy"]["value"]
    else:
        raise ValueError(f"没有已实现的全量精度协议: {model}.")
    if not math.isfinite(value) or value < 0 or (metric != "WER" and value > 1):
        raise ValueError("精度指标无效.")
    # 原始任务报告已经验证同协议参考时直接保留;其他任务只接受独立全量参考.
    reference = report.get("reference_accuracy")
    source = report.get("reference_source")
    reference_path = ctx.root / "models/onnx_reference.json"
    if model not in ("yolov5s", "yolov8n", "yamnet") and reference_path.is_file():
        baseline = read_json(reference_path)
        if (baseline.get("evaluation_scope") == "full"
                and baseline.get("model") == model
                and baseline.get("metric") == metric
                and baseline.get("samples") == samples
                and baseline.get("status") == "complete"):
            reference, source = baseline["accuracy"], baseline["source"]
    if reference is not None and (
            not math.isfinite(reference) or reference < 0 or (metric != "WER" and reference > 1)):
        raise ValueError("参考指标无效.")
    change = None if reference is None else (
        reference - value if metric == "WER" else value - reference) * 100
    return {
        "dataset": report.get("dataset", "coco_wholebody_val2017"),
        "samples": samples, "metric": metric, "board_accuracy": value,
        "reference_accuracy": reference, "reference_source": source,
        "accuracy_change_percentage_points": change,
        "comparison_status": "matched_reference" if reference is not None else "reference_unavailable",
        "raw_report": str(report_path.relative_to(ctx.root)),
    }


def llm_accuracy(ctx):
    """核对完整 MMLU/PPL 协议及模型标识,不读取 Demo 或性能文件."""
    data = ctx.root / "evaluation/full"
    mmlu = read_json(ctx.result / "mmlu/mmlu_accuracy.json")
    ppl = read_json(ctx.result / "wikitext/npu_quality.json")
    reference_mmlu = read_json(data / "reference/mmlu/mmlu_accuracy.json")
    reference_ppl = read_json(data / "quality/pytorch_quality.json")
    for report in (mmlu, ppl, reference_mmlu, reference_ppl):
        if report.get("evaluation_scope") != "full" or report.get("status") != "complete":
            raise ValueError("完整 LLM 评价不能使用子集或未完成报告.")
    for actual, reference, fields in (
        (mmlu, reference_mmlu, ("cases_sha256", "selected_source_question_ids", "evaluated_questions",
                               "scoring", "chat_template", "add_special_tokens", "context")),
        (ppl, reference_ppl, ("protocol", "selected_text_sha256", "corpus_sha256", "input_ids",
                             "blocks", "scored_tokens", "excluded_tail_tokens", "reset_context_each_block")),
    ):
        for field in fields:
            if actual[field] != reference[field]:
                raise ValueError(f"LLM 参考与硬件评价协议不一致: {field}.")
    require_count(mmlu["evaluated_questions"], 14042)
    if mmlu["scope"] != "full_test" or ppl["protocol"] != "full_test_block128_v1":
        raise ValueError("要求完整 MMLU 和 WikiText2 test 协议.")
    manifest = read_json(ctx.root / "manifest.json")
    prompt_name = f"{manifest['context']}c/prompt.dla"
    prompt = next(row for row in manifest["files"] if row["name"] == prompt_name)
    hardware = read_json(ctx.result / "mmlu/run_manifest.json")
    if hardware["graph_sha256"] != prompt["sha256"] or ppl["dla_sha256"] != prompt["sha256"]:
        raise ValueError("评价 DLA 与已有部署清单不一致.")
    if not all(math.isfinite(row["accuracy"]) and 0 <= row["accuracy"] <= 1
               for row in (mmlu, reference_mmlu)):
        raise ValueError("MMLU Accuracy 无效.")
    if not all(math.isfinite(row["perplexity"]) and row["perplexity"] > 0
               for row in (ppl, reference_ppl)):
        raise ValueError("WikiText2 PPL 无效.")
    return {"full_benchmark_evaluated": True, "accuracy": {
        "mmlu": {"questions": mmlu["evaluated_questions"],
                 "fp32_accuracy": reference_mmlu["accuracy"], "npu_accuracy": mmlu["accuracy"],
                 "difference_percentage_points": 100 * (mmlu["accuracy"] - reference_mmlu["accuracy"])},
        "wikitext": {"blocks": ppl["blocks"], "scored_tokens": ppl["scored_tokens"],
                     "protocol": ppl["protocol"], "fp32_ppl": reference_ppl["perplexity"],
                     "npu_ppl": ppl["perplexity"],
                     "difference": ppl["perplexity"] - reference_ppl["perplexity"]}}}


def main():
    """按独立运行编号写精度汇总,不清理预测或更新已归档指标."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if is_test():
        parser.error("本工具仅由独立全量评价入口调用.")
    ctx = paths(args.config.resolve().parent, args.config, args.run_id)
    print(f"[全量精度] 核对覆盖与协议: {ctx.run_id}.", flush=True)
    result = llm_accuracy(ctx) if ctx.config["name"] == "minicpm5_2b" else vision_accuracy(ctx)
    summary = {"model": ctx.config["name"], "run_id": ctx.run_id, "status": "complete",
               "evaluation_scope": "full", "purpose": "accuracy_only", **result}
    artifacts = [*ctx.root.glob("models/*.dla"), *ctx.root.glob("*c/*.dla")]
    summary["model_files"] = [{"file": str(path.relative_to(ctx.root)), "bytes": path.stat().st_size}
                              for path in artifacts]
    (ctx.result / "accuracy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[全量精度完成] {ctx.result / 'accuracy_summary.json'}; 预测和协议保留在本次结果目录.",
          flush=True)


if __name__ == "__main__":
    main()
