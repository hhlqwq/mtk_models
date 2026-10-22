"""按上传的同协议输入执行 MMLU 与 WikiText2 真实 MDLA 评价."""

import os
import json
import resource
from types import SimpleNamespace

from tools.board import context
from tools.evaluation import is_test
from tools.llm import evaluate_board_quality, evaluate_mmlu


def main():
    """运行当前范围的精度评价,全量不依赖生成或性能记录."""
    ctx = context(__file__)
    mode = "test" if is_test() else "full"
    data = ctx.root / "evaluation" / mode
    for name in ("mmlu/protocol.json", "mmlu/cases.json", "mmlu/board_input_contract.json",
                 "quality/board_quality_protocol.json", "quality/board_input_contract.json",
                 "reference/mmlu/mmlu_accuracy.json", "quality/pytorch_quality.json"):
        if not (data / name).is_file():
            raise FileNotFoundError(f"缺少评价资源: {data / name}; 请单独准备对应范围的输入.")
    library = ctx.root / "board/libneuron_bridge.so"
    if not library.is_file():
        raise FileNotFoundError(f"缺少已部署的 AArch64 桥接库: {library}.")
    ctx.result.mkdir(parents=True, exist_ok=True)
    evaluate_mmlu.evaluate(SimpleNamespace(
        backend="npu", data=data / "mmlu", package=ctx.root,
        library=library, output=ctx.result / "mmlu", diagnostic_questions=0))
    # 快速输入本身只有四块,不额外导出大型逐位置 logits.
    evaluate_board_quality.evaluate(SimpleNamespace(
        package=ctx.root, quality_data=data / "quality", library=library,
        output=ctx.result / "wikitext", diagnostic_blocks=0, decode_steps=0))
    memory = {"peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "scope": "mmlu_and_wikitext_python_process"}
    (ctx.result / "evaluation_memory.json").write_text(json.dumps(memory, indent=2) + "\n")
    print(f"[板端评价完成] {ctx.result}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    main()
