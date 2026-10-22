"""将已编译的原生 DeepSeek 模型整理为 NAS 同类目录结构."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

from transformers import AutoTokenizer
import yaml

from tools.files import copy_public_tools, sha256, single_file
from tools.deploy import configure_parser
from tools.storage import REPO_ROOT
from tools.llm.package import graph_contract, write_compile_scripts
from tools.summary import publish


def package(work, output, prefill, quantized_prefix=None):
    """复制真实 DLA 与量化资源,生成配置和待板端验证清单."""
    preparation = json.loads((work / "native_prepare.json").read_text())
    context = preparation["context"]
    model = work / "DeepSeek-R1-Distill-Qwen-1.5B"
    config = json.loads((model / "config.json").read_text())
    settings_path = work / "quantization_settings.json"
    saved_settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    prefix = saved_settings.get("quantized_prefix", "DeepSeek-R1-Distill-Qwen-1.5B_asym4W_sym16A_Overall_hessian")
    base = quantized_prefix or work / "tflite" / prefix
    if base.resolve().parent != (work / "tflite").resolve():
        raise ValueError("量化目录必须位于本次任务的 tflite 目录.")
    settings_path = work / "quantization_settings.json"
    if settings_path.exists():
        precision_settings = json.loads(settings_path.read_text())
        if precision_settings["quantized_prefix"] != base.name:
            raise ValueError("量化精度记录与打包目录不一致.")
        precision_values = [precision_settings[key]
                            for key in ("attention", "mlp", "lm_head")]
        if any(value != "asym4W_sym16A" for value in precision_values):
            raise ValueError("本项目交付仅允许全 W4A16.")
        precision = "asym4W_sym16A"
    else:
        # 兼容现有 W4 包,并核对量化目录的精度标识.
        if "_asym4W_sym16A_Overall_hessian" not in base.name:
            raise ValueError("新精度包缺少量化成功记录.")
        precision = "asym4W_sym16A"
        precision_settings = dict.fromkeys(("attention", "mlp", "lm_head"), precision)
    adapter_path = work / "quantization_adapter.json"
    adapter = json.loads(adapter_path.read_text()) if adapter_path.exists() else None
    # 归档统一精度的实际逐层核对记录,并核对其与交付配置一致.
    uniform_path = work / "weight_precision_audit.json"
    uniform_audit = (json.loads(uniform_path.read_text())
                     if uniform_path.exists() else None)
    if uniform_audit is not None:
        if any(uniform_audit[key] != precision_settings[key]
               for key in ("attention", "mlp", "lm_head")):
            raise ValueError("统一权重精度核对与交付配置不一致.")
        if uniform_audit["parts"] != config["num_hidden_layers"] + 1:
            raise ValueError("统一权重精度核对的分片数量不符.")
    prompt = single_file(Path(f"{base}_{prefill}t{context}c"), "*.dla")
    decode = single_file(Path(f"{base}_1t{context}c"), "*.dla")
    contracts = {"prompt": graph_contract(prompt.with_suffix(".tflite")),
                 "decode": graph_contract(decode.with_suffix(".tflite"))}
    embedding = single_file(base, "embedding_int16.bin")
    expected_bytes = config["hidden_size"] * config["vocab_size"] * 2
    if embedding.stat().st_size != expected_bytes:
        raise ValueError("原生 INT16 embedding 大小与官方词表不符.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("交付目录必须为空,避免覆盖既有交付文件.")
    for name in (f"{context}c", "tokenizer", "scripts/prompts"):
        (output / name).mkdir(parents=True, exist_ok=True)
    for src, dst in ((prompt, f"{context}c/prompt.dla"),
                     (decode, f"{context}c/decode.dla"),
                     (embedding, "tokenizer/embedding_int16.bin")):
        print(f"[打包] {dst}", flush=True)
        shutil.copyfile(src, output / dst)
    shutil.copyfile(REPO_ROOT / "tools/llm/evaluate_native.py",
                    output / "scripts/evaluate_native.py")
    tokenizer = json.loads((model / "tokenizer.json").read_text())
    with (output / "tokenizer/vocab.txt").open("wb") as stream:
        for text, index in tokenizer["model"]["vocab"].items():
            stream.write(text.encode("utf-8") + f"\n{index}\n".encode())
    merges = [" ".join(item) if isinstance(item, list) else item
              for item in tokenizer["model"]["merges"]]
    (output / "tokenizer/merges.txt").write_text("#version\n" + "\n".join(merges) + "\n")
    added = {item["id"]: item["content"] for item in tokenizer["added_tokens"]}
    (output / "tokenizer/added_tokens.yaml").write_text(
        yaml.safe_dump(added, allow_unicode=True))
    for path in (work / "prompts").glob("*.txt"):
        shutil.copyfile(path, output / "scripts/prompts" / path.name)
    options = {
        "promptTokenBatchSize": prefill, "genTokenBatchSize": 1,
        "cacheSize": context, "hiddenSize": config["hidden_size"],
        "numHead": config["num_attention_heads"],
        "numLayer": config["num_hidden_layers"], "maxTokenLength": context,
        "rotEmbBase": config["rope_theta"],
        "headDim": config["hidden_size"] // config["num_attention_heads"],
        "rotEmbNumInputs": 1, "splitMask": False,
        "modelInputType": "INT16", "modelOutputType": "INT16",
        "cacheType": "INT16", "maskType": "INT16", "rotEmbType": "INT16",
    }
    runtime = {
        "specialTokens": {"bosId": preparation["bos_token_id"],
                          "eosId": preparation["eos_token_id"], "addBos": False,
                          "stopToken": [preparation["eos_token_id"]]},
        "tokenizerPath": [f"./tokenizer/{name}" for name in
                          ("merges.txt", "vocab.txt", "added_tokens.yaml")],
        "tokenEmbPath": "./tokenizer/embedding_int16.bin",
        "dlaPromptPaths": [f"./{context}c/prompt.dla"],
        "dlaGenPaths": [f"./{context}c/decode.dla"],
    }
    (output / "scripts/config-yocto.yaml").write_text(
        yaml.safe_dump({"modelOptions": options, "runtimeOptions": runtime},
                       sort_keys=False, allow_unicode=True))
    write_compile_scripts(output)
    copy_public_tools(output / "scripts/tools", ("files", "evaluation", "board", "process", "summary"))
    shutil.copyfile(model / "LICENSE", output / "LICENSE")
    (output / "README.md").write_text(
        f"# DeepSeek-R1-Distill-Qwen-1.5B\n\n"
        f"Genio720 原生 {precision},Prefill {prefill},上下文 {context},Decode 1.\n"
        "当前打包状态为 compiled_pending_board,板端结果需另行记录.\n\n"
        "将整个目录复制到板端,执行 `bash compile.sh`.\n"
        "此步骤只保存展示示例;正式指标由 `bash evaluate.sh` 写入唯一 results/summary.json.\n"
        "Demo 使用官方 tokenizer 的 Token 输入,Prompt 文本同时保留供核对.\n"
        "16 条自编双语校准输入仅验证部署流程,不代表正式质量基准.\n")
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            print(f"[哈希] {path.relative_to(output)}", flush=True)
            files.append({"name": path.relative_to(output).as_posix(),
                          "bytes": path.stat().st_size, "sha256": sha256(path)})
    (output / "manifest.json").write_text(json.dumps({
        "model": "deepseek_r1_distill_qwen_1_5b",
        "status": "compiled_pending_board", "preparation": preparation,
        "precision": precision, "precision_settings": precision_settings,
        "quantization_adapter": adapter,
        "weight_precision_audit": uniform_audit,
        "weight_optimization": precision_settings.get(
            "weight_optimization",
            "hessian" if "_Overall_hessian" in base.name else
            "gradient" if "_Overall_gradient" in base.name else None),
        "optimized_layers": precision_settings.get("optimized_layers"),
        "optimization_calibration_samples": precision_settings.get("calibration_samples"),
        "cumulative_layer_error": precision_settings.get("cumulative_layer_error"),
        "prefill": prefill, "context": context, "files": files,
        "graph_contracts": contracts,
        "toolkit_sha256": "da10e770e2950542ab17c63182b0b932348ab05ffdceb18cbb097555ca6127f5",
    }, ensure_ascii=False, indent=2) + "\n")


def record_board_results(work, output, board_results):
    """只归档 evaluate.sh 已生成的唯一报告,不复制日志或自行拼接指标."""
    report_path = board_results / "summary.json"
    if not report_path.is_file():
        raise FileNotFoundError("请提供 evaluate.sh 生成的 results/summary.json.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if report.get("precision") != manifest["precision"]:
        raise ValueError("评价报告精度与当前交付模型不一致.")
    if report.get("context") != manifest["context"]:
        raise ValueError("评价报告上下文与当前交付模型不一致.")
    if report.get("status") not in ("complete", "partial") or not report.get("run_id"):
        raise ValueError("评价报告未完成或缺少运行编号.")
    publish(SimpleNamespace(root=output, result=board_results), report)


def main():
    """解析编译工作目录和新的交付目录,不覆盖已有包."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill", type=int, default=128)
    parser.add_argument("--quantized-prefix", type=Path)
    parser.add_argument("--board-results", type=Path)
    configure_parser(parser, __file__, {
        "work": lambda c: c.output / c.config["native"]["work_dir"],
        "output": lambda c: c.output / "models",
        "prefill": lambda c: c.config["native"]["prefill"],
    })
    args = parser.parse_args()
    if args.board_results:
        record_board_results(args.work, args.output, args.board_results)
    else:
        package(args.work, args.output, args.prefill, args.quantized_prefix)


if __name__ == "__main__":
    main()
