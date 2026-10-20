"""检查完整题集长度,生成同范围 FP32 参考及板端评价资源."""

import argparse
import os
from pathlib import Path
import re
from types import SimpleNamespace

from tools.deploy import context
from tools.evaluation import is_test
from tools.files import require_file, sha256, single_file
from tools.llm import evaluate_mmlu, evaluate_quality
from tools.llm.native import settings


def evaluate_mode(native, prepare_only):
    """准备当前模式的输入和参考,同一次流水线使用独立运行目录."""
    config = native.ctx.config
    if native.prefill != 128 or native.context % 128:
        raise ValueError("当前评价协议要求 Prefill=128,上下文为 128 的整数倍.")
    mode = "test" if is_test() else "full"
    run_id = os.environ["EVAL_RUN_ID"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("运行编号只允许字母、数字、下划线和连字符.")
    output = native.work / "evaluation" / run_id / mode
    data = config["data"]
    mmlu = require_file(native.ctx.dataset / data["mmlu_test"])
    corpus = require_file(native.ctx.dataset / data["wikitext_test"])
    if sha256(corpus) != "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91":
        raise ValueError("WikiText2 test 哈希不符,量化前停止.")
    graph = None if prepare_only else single_file(
        Path(f"{native.quantized}_{native.prefill}t{native.context}c"), "*.tflite")
    mmlu_args = SimpleNamespace(
        dataset=mmlu, model=native.model, graph=graph, context=native.context,
        limit=0, seed=17, output=output / "mmlu",
        test_questions=config["evaluation"]["test_mmlu_questions"])
    print(f"[评价范围] {mode}; MMLU 题集长度检查在量化前执行.", flush=True)
    evaluate_mmlu.prepare(mmlu_args)
    if prepare_only:
        return
    mmlu_args = SimpleNamespace(
        backend="pytorch", data=output / "mmlu", model=native.model,
        output=output / "reference/mmlu", diagnostic_questions=0)
    evaluate_mmlu.evaluate(mmlu_args)
    quality_args = SimpleNamespace(
        model=native.model, corpus=corpus, output=output / "quality",
        backend="pytorch", tflite=graph.parent, decode_tflite=None,
        text_rows=128, full_test=not is_test(), blocks=None,
        diagnostic_blocks=config["evaluation"]["test_wikitext_blocks"] if is_test() else 0,
        decode_steps=0, skip_demo=True,
        device=config["evaluation"]["reference_device"], cpu_threads=8)
    evaluate_quality.evaluate(quality_args)
    quality_args.backend = "prepare_board"
    evaluate_quality.evaluate(quality_args)
    print(f"[评价资源完成] {output}", flush=True)


def main():
    """全量部署同时携带快速参考,快速部署只准备固定小子集."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--config", type=Path)
    options = parser.parse_args()
    native = settings(context(__file__, options.config))
    original_mode = os.environ.get("MTK_EVAL_TEST", "0")
    modes = ("test",) if is_test() else ("full", "test")
    try:
        for mode in modes:
            os.environ["MTK_EVAL_TEST"] = "1" if mode == "test" else "0"
            evaluate_mode(native, options.prepare_only)
    finally:
        os.environ["MTK_EVAL_TEST"] = original_mode


if __name__ == "__main__":
    main()
