"""执行原生 CLI 生成及统一定长性能,保留每次运行的原始证据."""

import json

from tools.board import context
from tools.evaluation import is_test, record_selection, result_status, scope_metadata
from tools.llm.evaluate_native import run_sample


def main():
    """快速模式限制生成长度,固定性能协议保持 128 输入与 128 输出上限."""
    ctx = context(__file__)
    mode = "test" if is_test() else "full"
    if mode not in ctx.config["evaluation"]["prepared_modes"]:
        raise ValueError(f"部署包只准备了 {ctx.config['evaluation']['prepared_mode']} 评价数据,请上传 {mode} 包.")
    ctx.result.mkdir(parents=True, exist_ok=False)
    limit = 32 if is_test() else ctx.config["conversion"]["max_new_tokens"]
    report = {"run_id": ctx.run_id, "scope": "generation_smoke_and_fixed_performance",
              "samples": [], "text_input_checks": []}
    for language in ("zh", "en"):
        row, answer = run_sample(ctx.root, ctx.result, language, True, limit,
                                 require_eos=not is_test())
        check, text = run_sample(ctx.root, ctx.result, language, False, limit,
                                 require_eos=not is_test())
        check["text_equals_token_input_response"] = answer == text
        if answer != text:
            raise ValueError(f"文本与 Token 输入输出不同: {language}.")
        report["samples"].append(row)
        report["text_input_checks"].append(check)
    performance, unused_text = run_sample(
        ctx.root, ctx.result, "performance", True, 128,
        ctx.root / "scripts/prompts/performance_128.tokens.txt", require_eos=False)
    performance["scope"] = "fixed_128_input_max_128_generated_performance"
    report["performance"] = performance
    record_selection(("zh", "en"))
    report.update(status=result_status(), **scope_metadata())
    (ctx.result / "board_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"[生成与性能完成] {ctx.result}", flush=True)


if __name__ == "__main__":
    main()
