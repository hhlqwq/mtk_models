"""评价完成后统一发布精度、性能与峰值 RSS,results 仅保留 summary.json."""

import json
import math
import os
import platform
import shutil

from tools.board import context
from tools.evaluation import is_test
from tools.files import read_json


def require_count(actual, expected):
    """拒绝不完整覆盖,不把固定子集当作全量评价."""
    if actual != expected:
        raise ValueError(f"全量覆盖不完整: {actual}/{expected}.")


def vision_accuracy(ctx):
    """核对完整任务精度与匹配参考,同时纳入本次硬件性能."""
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
        "performance": vision_performance(ctx, report),
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


def vision_performance(ctx, report):
    """读取同一次硬件推理的性能和进程 RSS,不使用主机参考耗时."""
    model = ctx.config["name"]
    if model == "rtmpose_body2d":
        report = read_json(ctx.work / "timing_summary.json")
    if model == "whisper_tiny":
        overall = report["performance"]["overall"]
        mean = overall["npu_total_ms"]["mean"]
        p95 = overall["npu_total_ms"]["p95"]
        rss = overall["peak_rss_kb"]["max"] / 1024
        scope = "每条音频 Encoder 与全部 Decoder NPU 调用之和"
    else:
        mean = report["npu_mean_ms"]
        p95 = report.get("npu_p95_ms")
        rss = report.get("peak_rss_mib")
        if rss is None:
            value = report.get("peak_rss_kib", report.get("peak_rss_kb"))
            rss = None if value is None else value / 1024
        scope = report.get("timing_scope", "每个人体裁剪的 NPU 调用" if model == "rtmpose_body2d"
                           else "本次实际输入的 NPU 调用")
    if not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean <= 0:
        raise ValueError("缺少有效的板端 NPU 性能记录.")
    if not isinstance(rss, (int, float)) or not math.isfinite(rss) or rss <= 0:
        raise ValueError("缺少有效的推理进程峰值 RSS.")
    result = {"npu_mean_ms": mean, "npu_p95_ms": p95, "scope": scope,
              "peak_process_rss_mib": rss,
              "memory_scope": "推理进程峰值 RSS,不等于设备总内存或驱动专用内存"}
    for key in ("preprocess_mean_ms", "postprocess_mean_ms", "end_to_end_mean_ms",
                "preprocess_mean_ms_per_clip", "inference_stage_mean_ms_per_clip",
                "frontend_peak_rss_mib"):
        if key in report:
            result[key] = report[key]
    if model in ("yolov5s", "yolov8n"):
        timing = read_json(ctx.work / "timing_summary_current_run.json")
        result["npu_p95_ms"] = timing["npu_ms"].get("p95")
        result["end_to_end_ms"] = timing["e2e_ms"]
        result["end_to_end_scope"] = "常驻 C++ 程序的逐图预处理、推理和后处理"
    if result["npu_p95_ms"] is None:
        result["unavailable_metrics"] = ["npu_p95_ms: 当前运行器未记录"]
    return result


def llm_performance(ctx):
    """读取定长 Prefill/Decode 评价与评价进程 RSS,拒绝混入其他运行."""
    generation = read_json(ctx.result / "board_metrics.json")
    if generation["run_id"] != ctx.run_id:
        raise ValueError("LLM 性能记录不属于本次评价.")
    speed = generation["performance"]
    memory = read_json(ctx.result / "evaluation_memory.json")
    cli = [speed, *generation["samples"], *generation["text_input_checks"]]
    values = [row["peak_process_rss_mib"] for row in cli] + [memory["peak_process_rss_mib"]]
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("LLM 进程 RSS 记录无效.")
    return {"prefill_tokens_per_second": speed["prefill_tokens_per_second"],
            "decode_tokens_per_second": speed["native_reported_decode_tokens_per_second"],
            "scope": speed["scope"], "input_tokens": speed["prefill_input_tokens"],
            "generated_tokens": len(speed["generated_tokens"]),
            "peak_process_rss_mib": max(values),
            "memory_scope": "本次生成、定长测速和精度评价各进程峰值的最大值",
            "timing_scope": "原生 CLI 报告的 Prefill 与 Decode 阶段,不是纯 NPU 算子耗时",
            "unavailable_metrics": ["纯 NPU Prefill/Decode 分项耗时: 原生 CLI 未提供",
                                    "端到端生成速度: 未采用独立的统一计时协议"],
            "unavailable_checks": generation.get("unavailable_checks", []),
            "warmup_runs": generation.get("warmup_runs", 0),
            "measured_runs": generation.get("measured_runs", 1),
            "functional_failures": generation["functional_failures"]}


def publish(ctx, report):
    """发布唯一正式报告,仅清理当前部署包 results 内已由新报告取代的文件."""
    directory = ctx.root / "results"
    if directory.is_symlink() or not directory.resolve().is_relative_to(ctx.root.resolve()):
        raise ValueError("正式报告目录越出当前部署包.")
    directory.mkdir(exist_ok=True)
    previous = list(directory.iterdir())
    for path in previous:
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"结果目录含越界或链接文件,请先处理: {path.name}.")
    # 先完整序列化,无效数值不能损坏已有报告.
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination = directory / "summary.json"
    if destination.is_symlink():
        raise ValueError("正式报告不能是符号链接.")
    destination.write_text(text, encoding="utf-8")
    for path in previous:
        if path.name == "summary.json":
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"拒绝清理越界或链接文件: {path.name}.")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    print(f"[评价报告] {destination}; 过程诊断保留在 {ctx.result}.", flush=True)


def main():
    """只由 evaluate.sh 发布一个包含实际覆盖、精度、性能和 RSS 的报告."""
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    if is_test():
        raise ValueError("compile.sh 只生成示例,不允许发布正式评价报告.")
    ctx = context(__file__)
    model = ctx.config["name"]
    print(f"[评价汇总] 核对同次精度、性能与峰值 RSS: {ctx.run_id}.", flush=True)
    if model in ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b"):
        result = llm_accuracy(ctx)
        result["performance"] = llm_performance(ctx)
        result["precision"] = read_json(ctx.root / "manifest.json")["precision"]
        if model == "deepseek_r1_distill_qwen_1_5b":
            manifest = read_json(ctx.root / "manifest.json")
            result.update(context=manifest["context"], prefill=manifest["prefill"],
                          precision_settings=manifest["precision_settings"],
                          source={key: manifest["preparation"]["source"][key]
                                  for key in ("repo", "revision")})
        if result["performance"]["functional_failures"]:
            result["status"] = "functional_checks_failed"
    else:
        result = vision_accuracy(ctx)
    summary = {"schema_version": 1, "model": model, "run_id": ctx.run_id,
               "status": "complete", "evaluation_scope": "full", **result,
               "runtime": {"machine": platform.machine(), "kernel": platform.release()}}
    artifacts = [*ctx.root.glob("models/*.dla"), *ctx.root.glob("*c/*.dla"),
                 *ctx.root.glob("models/embedding*.bin"), *ctx.root.glob("tokenizer/embedding*.bin")]
    summary["model_files"] = [{"file": str(path.relative_to(ctx.root)), "bytes": path.stat().st_size}
                              for path in artifacts]
    publish(ctx, summary)


if __name__ == "__main__":
    main()
