"""检查题集长度,直接调用公共评价函数生成 FP32 参考和板端输入."""

import argparse
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

from tools.deploy import context
from tools.evaluation import is_test
from tools.files import require_file, sha256, single_file
from tools.llm import evaluate_mmlu, evaluate_quality
from tools.llm.native import settings
from tools.storage import storage_roots


def check_data(ctx):
    """在读取权重前检查 NAS 正式资源,不从临时目录回退或自动下载."""
    roots = storage_roots()
    dataset_root = roots["dataset_root"].resolve()
    models_root = roots["models_root"].resolve()
    temporary = (Path("/tmp"), Path("/var/tmp"))
    if any(root.is_relative_to(base) for root in (dataset_root, models_root)
           for base in temporary):
        raise ValueError("models_root 和 dataset_root 必须指向 NAS,不能使用临时目录.")
    source = Path(os.environ.get("MODEL_SOURCE", ctx.config["model"]["source"])).resolve()
    work = Path(os.environ.get(
        "NATIVE_WORK", ctx.output / ctx.config["native"]["work_dir"])).resolve()
    if not all(path.is_relative_to(models_root)
               for path in (source, work, ctx.output.resolve())):
        raise ValueError("官方模型和转换资源必须位于 NAS models_root 内.")
    manifest = require_file(source / "source_manifest.json")
    for entry in json.loads(manifest.read_text())["files"]:
        path = (source / entry["name"]).resolve()
        if not path.is_relative_to(models_root):
            raise ValueError(f"官方资源不能引用 NAS 之外的文件: {path}.")
        require_file(path)
    checks = (
        ("mmlu_test", evaluate_mmlu.DATA_SHA256),
        ("wikitext_test", "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"),
    )
    result = []
    for key, expected in checks:
        path = (ctx.dataset / ctx.config["data"][key]).resolve()
        if not path.is_relative_to(dataset_root):
            raise ValueError(f"{key} 必须位于 NAS dataset_root 内: {path}.")
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"NAS 缺少 {key}: {path}.请先准备正式数据,不会搜索 /tmp.")
        if sha256(path) != expected:
            raise ValueError(f"{key} 与固定版本不一致: {path}.")
        result.append(path)
        print(f"[资源就绪] {key}: {path}", flush=True)
    train = os.environ.get("CALIBRATION_TRAIN", ctx.config["native"]["calibration_train"])
    if train:
        path = Path(train).resolve()
        if not path.is_relative_to(dataset_root):
            raise ValueError("校准训练集必须使用 NAS dataset_root 内的绝对路径.")
        if sha256(require_file(path)) != "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7":
            raise ValueError("WikiText2 train 与固定版本不一致.")
    return result


def evaluate_mode(native, prepare_only, datasets):
    """准备当前模式的输入和参考,同一次流水线使用独立运行目录."""
    config = native.ctx.config
    if native.prefill != 128 or native.context % 128:
        raise ValueError("当前评价协议要求 Prefill=128,上下文为 128 的整数倍.")
    mode = "test" if is_test() else "full"
    run_id = os.environ["EVAL_RUN_ID"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("运行编号只允许字母、数字、下划线和连字符.")
    output = native.work / "evaluation" / mode
    mmlu, corpus = datasets
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
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--config", type=Path)
    options = parser.parse_args()
    ctx = context(__file__, options.config)
    datasets = check_data(ctx)
    if options.check_data:
        return
    native = settings(ctx)
    original_mode = os.environ.get("MTK_EVAL_TEST", "0")
    modes = ("test",) if is_test() else ("full", "test")
    try:
        for mode in modes:
            os.environ["MTK_EVAL_TEST"] = "1" if mode == "test" else "0"
            evaluate_mode(native, options.prepare_only, datasets)
    finally:
        os.environ["MTK_EVAL_TEST"] = original_mode


if __name__ == "__main__":
    main()
