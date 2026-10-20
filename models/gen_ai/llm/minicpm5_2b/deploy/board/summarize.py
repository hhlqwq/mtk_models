"""汇总同协议板端指标,快速结果与全量结果分别保存."""

import json
import math

from tools.board import context
from tools.evaluation import is_test, result_status
from tools.files import read_json, sha256


def compare_fields(reference, actual, fields):
    """要求参考与硬件的评价输入、范围和评分数量逐项相同."""
    for key in fields:
        if reference[key] != actual[key]:
            raise ValueError(f"FP32 与 NPU 协议不一致: {key}.")


def main():
    """只有全部评价完成且协议吻合才保存成功汇总,不自动宣布质量合格."""
    ctx = context(__file__)
    mode = "test" if is_test() else "full"
    data = ctx.root / "evaluation" / mode
    reference_mmlu = read_json(data / "reference/mmlu/mmlu_accuracy.json")
    reference_ppl = read_json(data / "quality/pytorch_quality.json")
    mmlu = read_json(ctx.result / "mmlu/mmlu_accuracy.json")
    ppl = read_json(ctx.result / "wikitext/npu_quality.json")
    generation = read_json(ctx.result / "board_metrics.json")
    manifest = read_json(ctx.root / "manifest.json")
    expected_scope = "subset" if is_test() else "full"
    for report in (reference_mmlu, reference_ppl, mmlu, ppl, generation):
        if report["evaluation_scope"] != expected_scope:
            raise ValueError("快速和全量结果混用,拒绝汇总.")
    compare_fields(reference_mmlu, mmlu, (
        "cases_sha256", "selected_source_question_ids", "evaluated_questions",
        "scoring", "chat_template", "add_special_tokens", "context"))
    compare_fields(reference_ppl, ppl, (
        "protocol", "selected_text_sha256", "corpus_sha256", "input_ids",
        "blocks", "scored_tokens", "excluded_tail_tokens", "reset_context_each_block"))
    if not is_test() and (mmlu["evaluated_questions"] != 14042
                          or mmlu["scope"] != "full_test"
                          or ppl["protocol"] != "full_test_block128_v1"):
        raise ValueError("默认入口必须完成全部 MMLU 与完整 WikiText2 test.")
    if is_test() and (mmlu["evaluated_questions"] != ctx.config["evaluation"]["test_mmlu_questions"]
                      or ppl["blocks"] != ctx.config["evaluation"]["test_wikitext_blocks"]):
        raise ValueError("快速评价数量与配置不符.")
    hardware = read_json(ctx.result / "mmlu/run_manifest.json")
    prompt_name = f"{manifest['context']}c/prompt.dla"
    prompt = next(row for row in manifest["files"] if row["name"] == prompt_name)
    if hardware["graph_sha256"] != prompt["sha256"] or ppl["dla_sha256"] != prompt["sha256"]:
        raise ValueError("评价使用的 DLA 与交付清单不一致.")
    for row in generation["samples"]:
        if sha256(ctx.result / f"{row['id']}.log") != row["log_sha256"]:
            raise ValueError("生成日志哈希不符.")
    if not all(check["text_equals_token_input_response"] for check in generation["text_input_checks"]):
        raise ValueError("生成输入一致性检查未通过.")
    values = (reference_ppl["perplexity"], ppl["perplexity"])
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("PPL 不是有效正数.")
    files = [row for row in manifest["files"] if row["name"].endswith(".dla")
             or row["name"] == "tokenizer/embedding_int16.bin"]
    sizes = [{"file": row["name"], "bytes": (ctx.root / row["name"]).stat().st_size}
             for row in files]
    if any(size["bytes"] != row["bytes"] for size, row in zip(sizes, files)):
        raise ValueError("部署模型大小与清单不一致.")
    speed = generation["performance"]
    if (speed["prefill_input_tokens"] != 128 or len(speed["generated_tokens"]) < 2
            or speed["prefill_seconds"] <= 0
            or not math.isfinite(speed["native_reported_decode_tokens_per_second"])
            or speed["native_reported_decode_tokens_per_second"] <= 0
            or sha256(ctx.result / "performance_demo.log") != speed["log_sha256"]):
        raise ValueError("统一性能计数、速度或原始日志无效.")
    summary = {
        "model": "minicpm5_2b", "run_id": ctx.run_id,
        "status": result_status("board_verified"), "evaluation_scope": expected_scope,
        "quality_qualified": False,
        "quality_note": "运行完成不代表精度合格,须按验收要求审阅 FP32/NPU 差值.",
        "full_benchmark_evaluated": not is_test(), "precision": manifest["precision"],
        "context": manifest["context"], "prefill": manifest["prefill"],
        "accuracy": {
            "mmlu": {"questions": mmlu["evaluated_questions"],
                     "fp32_accuracy": reference_mmlu["accuracy"], "npu_accuracy": mmlu["accuracy"],
                     "difference_percentage_points": 100 * (mmlu["accuracy"] - reference_mmlu["accuracy"]),
                     "cases_sha256": mmlu["cases_sha256"]},
            "wikitext": {"blocks": ppl["blocks"], "scored_tokens": ppl["scored_tokens"],
                         "protocol": ppl["protocol"], "fp32_ppl": values[0], "npu_ppl": values[1],
                         "relative_increase_percent": 100 * (values[1] / values[0] - 1)}},
        "performance_summary": {
            "unit": "token/s", "scope": speed["scope"],
            "prefill_tokens_per_second": speed["prefill_tokens_per_second"],
            "decode_tokens_per_second": speed["native_reported_decode_tokens_per_second"],
            "input_tokens": speed["prefill_input_tokens"],
            "generated_tokens": len(speed["generated_tokens"])},
        "model_size": {"unit": "MB", "deployment_total": round(sum(row["bytes"] for row in sizes) / 1e6, 2),
                       "files": sizes}, "cpu_fallback": False,
        "evidence": str(ctx.result.relative_to(ctx.root))}
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    (ctx.result / "summary.json").write_text(text)
    latest = ctx.root / "results" / ("summary_test.json" if is_test() else "summary.json")
    latest.write_text(text)
    print(f"[汇总完成] {latest}; 范围={expected_scope}; quality_qualified=false.", flush=True)


if __name__ == "__main__":
    main()
