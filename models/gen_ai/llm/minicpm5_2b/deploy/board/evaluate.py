"""按上传的同协议输入执行 MMLU 与 WikiText2 真实 MDLA 评价."""

import json
import resource
from types import SimpleNamespace

from tools.board import context
from tools.evaluation import is_test
from tools.llm import evaluate_board_quality, evaluate_mmlu


def main():
    """默认运行完整题集和文本,快速模式只运行预先固定的小子集."""
    ctx = context(__file__)
    mode = "test" if is_test() else "full"
    if mode not in ctx.config["evaluation"]["prepared_modes"]:
        raise ValueError("部署包评价范围与执行模式不一致,请重新准备并上传对应模式.")
    data = ctx.root / "evaluation" / mode
    library = ctx.root / "board/libneuron_bridge.so"
    if not ctx.result.is_dir():
        raise ValueError("缺少本次生成记录,请通过 run.sh 一键执行.")
    evaluate_mmlu.evaluate(SimpleNamespace(
        backend="npu", data=data / "mmlu", package=ctx.root,
        library=library, output=ctx.result / "mmlu", diagnostic_questions=0))
    # 快速输入本身只有四块,不额外导出大型逐位置 logits.
    evaluate_board_quality.evaluate(SimpleNamespace(
        package=ctx.root, quality_data=data / "quality", library=library,
        output=ctx.result / "wikitext", diagnostic_blocks=0, decode_steps=0))
    memory = {"peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "scope": "mmlu_and_wikitext_python_process"}
    (ctx.result / "evaluation_memory.json").write_text(
        json.dumps(memory, indent=2) + "\n")
    print(f"[评价进程峰值 RSS] {memory['peak_process_rss_mib']:.2f} MiB.", flush=True)
    print(f"[板端评价完成] {ctx.result}", flush=True)


if __name__ == "__main__":
    main()
