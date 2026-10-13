"""将已编译的原生 DeepSeek 模型整理为 NAS 同类目录结构."""

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml

from prepare_native import sha256


def graph_contract(path):
    """读取原生静态图 I/O,要求配置声明的所有接口实际为 INT16."""
    from mtk_converter.python.converters.tflite.schema.tflite.Model import Model
    from mtk_converter.python.converters.tflite.schema.tflite.TensorType import TensorType

    graph = Model.GetRootAsModel(path.read_bytes(), 0).Subgraphs(0)
    result = {"inputs": [], "outputs": []}
    checked = 0
    for index in range(graph.TensorsLength()):
        tensor = graph.Tensors(index)
        quantization = tensor.Quantization()
        if quantization is not None:
            for offset in range(quantization.ScaleLength()):
                scale = quantization.Scale(offset)
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError(f"量化 Scale 非法: {tensor.Name()},scale={scale}")
                checked += 1
    result["quantization_scales_checked"] = checked
    for kind, length, index in (("inputs", graph.InputsLength, graph.Inputs),
                                ("outputs", graph.OutputsLength, graph.Outputs)):
        for offset in range(length()):
            tensor = graph.Tensors(index(offset))
            if tensor.Type() != TensorType.INT16:
                raise ValueError(f"原生图包含非 INT16 的 {kind}: {tensor.Name()}")
            result[kind].append({"name": tensor.Name().decode(), "dtype": "INT16",
                                 "shape": tensor.ShapeAsNumpy().tolist()})
    return result


def single_file(folder, pattern):
    """要求编译目录中存在唯一且非空的目标文件."""
    matches = list(folder.glob(pattern))
    if len(matches) != 1 or matches[0].stat().st_size == 0:
        raise ValueError(f"期望唯一非空文件: {folder}/{pattern}")
    return matches[0]


def write_run_script(output):
    """生成已验证的 Token/文本输入入口,正确处理原生 CLI 的生成计数."""
    (output / "scripts/run.sh").write_text(
        '#!/usr/bin/env bash\n'
        '# 运行原生双语 Demo,保留实际输出与运行日志.\n'
        'set -euo pipefail\n'
        'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
        'cd "${ROOT}"\n'
        'RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"\n'
        '[[ "${RUN_ID}" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2\n'
        'MAX_TOKENS="${MAX_NEW_TOKENS:-768}"\n'
        '[[ "${MAX_TOKENS}" =~ ^[0-9]+$ ]] && (( MAX_TOKENS >= 2 )) || exit 2\n'
        '# 原生 CLI 的 -m 统计首 Token 后的调用次数,总生成上限需减 1.\n'
        'case "${INPUT_MODE:-tokens}" in\n'
        '    tokens) TOKEN_OPTION=(--read-tokens); SUFFIX=.tokens.txt ;;\n'
        '    text) TOKEN_OPTION=(); SUFFIX=.txt ;;\n'
        '    *) echo "[错误] INPUT_MODE 必须为 tokens 或 text." >&2; exit 2 ;;\n'
        'esac\n'
        'mkdir -p "results/${RUN_ID}"\n'
        'for language in zh en; do\n'
        '    echo "[板端] ${language} Demo"\n'
        '    llm_cmdline_tool scripts/config-yocto.yaml "${TOKEN_OPTION[@]}" '
        '-i "scripts/prompts/${language}_demo${SUFFIX}" '
        '-m "$((MAX_TOKENS - 1))" 2>&1 | '
        'tee "results/${RUN_ID}/${language}_demo.log"\n'
        'done\n')


def package(work, output, prefill, quantized_prefix=None):
    """复制真实 DLA 与量化资源,生成配置和待板端验证清单."""
    preparation = json.loads((work / "native_prepare.json").read_text())
    context = preparation["context"]
    model = work / "DeepSeek-R1-Distill-Qwen-1.5B"
    config = json.loads((model / "config.json").read_text())
    base = quantized_prefix or work / "tflite/DeepSeek-R1-Distill-Qwen-1.5B_asym4W_sym16A_Overall_hessian"
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
    shutil.copyfile(Path(__file__).resolve().parents[6] / "tools/llm/evaluate_native.py",
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
    write_run_script(output)
    shutil.copyfile(model / "LICENSE", output / "LICENSE")
    (output / "README.md").write_text(
        f"# DeepSeek-R1-Distill-Qwen-1.5B\n\n"
        f"Genio720 原生 {precision},Prefill {prefill},上下文 {context},Decode 1.\n"
        "当前打包状态为 compiled_pending_board,板端结果需另行记录.\n\n"
        "将整个目录复制到板端,执行 `bash scripts/run.sh`.\n"
        "Demo 使用官方 tokenizer 的 Token 输入,Prompt 文本同时保留供核对.\n"
        "16 条自编双语校准输入仅验证部署流程,不代表正式质量基准.\n")
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            print(f"[哈希] {path.relative_to(output)}", flush=True)
            files.append({"name": path.relative_to(output).as_posix(),
                          "bytes": path.stat().st_size, "sha256": sha256(path)})
    (output / "manifest.json").write_text(json.dumps({
        "status": "compiled_pending_board", "preparation": preparation,
        "precision": precision, "precision_settings": precision_settings,
        "quantization_adapter": adapter,
        "weight_precision_audit": uniform_audit,
        "weight_optimization": "hessian",
        "prefill": prefill, "context": context, "files": files,
        "graph_contracts": contracts,
        "toolkit_sha256": "da10e770e2950542ab17c63182b0b932348ab05ffdceb18cbb097555ca6127f5",
    }, ensure_ascii=False, indent=2) + "\n")


def record_board_results(work, output, results):
    """验证原始板端证据后更新交付状态,不把运行成功视为正式质量达标."""
    from transformers import AutoTokenizer

    manifest = json.loads((output / "manifest.json").read_text())
    for entry in manifest["files"]:
        path = output / entry["name"]
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"交付文件已变化: {path}")
    metrics = json.loads((results / "board_metrics.json").read_text())
    if not re.fullmatch(r"[A-Za-z0-9_-]+", metrics["run_id"]):
        raise ValueError("板端运行编号无效.")
    if {row["id"] for row in metrics["samples"]} != {"zh_demo", "en_demo"}:
        raise ValueError("需要中英文各一个原始板端结果.")
    checks = metrics.get("text_input_checks", [])
    if len(checks) != 2 or not all(
            row["text_equals_token_input_response"] for row in checks):
        raise ValueError("尚未通过原生文本输入一致性检查.")
    eos = manifest["preparation"]["eos_token_id"]
    tokenizer = AutoTokenizer.from_pretrained(
        work / "DeepSeek-R1-Distill-Qwen-1.5B", local_files_only=True)
    for row in metrics["samples"]:
        if row["returncode"] != 0 or not row["eos_reached"] or row["generated_tokens"][-1] != eos:
            raise ValueError(f"板端回答未完成: {row['id']}")
        if sha256(results / f"{row['id']}.log") != row["log_sha256"]:
            raise ValueError("原始板端日志哈希不符.")
        row["generated_token_count"] = len(row["generated_tokens"])
        prompt_path = output / "scripts/prompts" / f"{row['id']}.tokens.txt"
        row["prefill_input_tokens"] = len(re.findall(r"\d+", prompt_path.read_text()))
        if row["prefill_seconds"] <= 0 or row["prefill_input_tokens"] == 0:
            raise ValueError("Prefill 输入数量或耗时无效.")
        row["prefill_tokens_per_second"] = (
            row["prefill_input_tokens"] / row["prefill_seconds"])
        row["text"] = tokenizer.decode(
            row["generated_tokens"], skip_special_tokens=False,
            clean_up_tokenization_spaces=False)
        final = row["text"].split("</think>", 1)
        row["final_answer_present"] = len(final) == 2 and bool(
            final[-1].replace(tokenizer.eos_token, "").strip())
        if not row["final_answer_present"]:
            raise ValueError(f"缺少最终回答: {row['id']}")
    for check in checks:
        name = check["id"].replace("_demo", "_text_input") + ".log"
        if sha256(results / name) != check["log_sha256"]:
            raise ValueError("原始文本输入检查日志哈希不符.")
    destination = output / "results" / metrics["run_id"]
    destination.mkdir(parents=True, exist_ok=False)
    for path in results.iterdir():
        if path.is_file() and path.suffix in (".log", ".json"):
            shutil.copyfile(path, destination / path.name)
    metrics.update(status="board_verified", precision=manifest["precision"],
                   prefill=manifest["prefill"], context=manifest["context"],
                   formal_accuracy_evaluated=False,
                   qualitative_observations=["英文样例出现不自然的语法,正式量化质量尚未验收."],
                   memory_scope="process_vm_hwm_and_sampled_system_memavailable")
    # Prefill 使用总输入数与总耗时,Decode 对 CLI 速度做 Token 加权调和汇总.
    input_tokens = sum(row["prefill_input_tokens"] for row in metrics["samples"])
    prompt_seconds = sum(row["prefill_seconds"] for row in metrics["samples"])
    generated = sum(len(row["generated_tokens"]) for row in metrics["samples"])
    decode_weight = sum(len(row["generated_tokens"]) /
                        row["native_reported_decode_tokens_per_second"]
                        for row in metrics["samples"])
    metrics["performance_summary"] = {
        "unit": "token/s", "prefill_tokens_per_second": input_tokens / prompt_seconds,
        "decode_tokens_per_second": generated / decode_weight,
        "prefill_input_tokens": input_tokens, "prefill_seconds": prompt_seconds,
        "decode_aggregation": "generated_token_weighted_harmonic_mean_of_cli_rates",
        "scope": "fixed_demo_samples_not_full_benchmark",
    }
    performance = metrics.get("performance")
    if performance is not None:
        if performance["returncode"] != 0 or performance["prefill_input_tokens"] != 128 or \
                len(performance["generated_tokens"]) < 2 or \
                performance["native_reported_decode_tokens_per_second"] <= 0 or \
                sha256(results / "performance_demo.log") != performance["log_sha256"]:
            raise ValueError("统一性能结果或原始日志不符.")
        metrics["performance_summary"] = {
            "unit": "token/s", "scope": performance["scope"],
            "prefill_tokens_per_second": performance["prefill_tokens_per_second"],
            "decode_tokens_per_second": performance["native_reported_decode_tokens_per_second"],
            "prefill_input_tokens": 128, "prefill_seconds": performance["prefill_seconds"],
            "generated_token_count": len(performance["generated_tokens"]),
            "decode_aggregation": "native_cli_fixed_prompt",
        }
    # 只统计交付目录中的文件,不执行模型或重复质量评测.
    metrics["model_size"] = json.loads(subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve().parents[6] /
                              "tools/model_size.py"),
         "--models-dir", str(output), "--include", "1024c/*.dla",
         "--include", "tokenizer/embedding*.bin",
         "--source", "package_file_stat"], text=True))
    (output / "results/summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    write_run_script(output)
    readme = (output / "README.md").read_text().replace(
        "当前打包状态为 compiled_pending_board,板端结果需另行记录.",
        "当前状态为 board_verified,仅表示板端推理与双语样例完成,正式质量待评测.")
    readme += "\n默认总生成上限 768 Token.执行 `INPUT_MODE=text bash scripts/run.sh` 可使用原生文本分词.\n"
    readme += "两个固定样例的文本输入结果与官方 Token 输入结果一致.\n\n"
    performance = metrics["performance_summary"]
    readme += "| Prefill token/s | Decode token/s |\n| ---: | ---: |\n"
    readme += (f"| {performance['prefill_tokens_per_second']:.4f} | "
               f"{performance['decode_tokens_per_second']:.4f} |\n")
    if metrics.get("performance") is not None:
        readme += "固定 128 Token 输入,最多生成 128 Token; Decode 使用原生 CLI 计数,不按语言分组.\n"
    else:
        readme += "Prefill 按总输入 Token/总 Prompt 耗时汇总; Decode 为 CLI 速度的生成 Token 加权调和平均.\n"
    readme += "\n英文样例存在不自然的语法,原始日志与 Token 均保留,不能将本次结果视为正式精度达标.\n"
    readme += "Prefill 时间不包含加载和模型切换,不作为 TTFT; Runtime 速度按原生 CLI 的 Decode 计数.\n"
    (output / "README.md").write_text(readme)
    manifest.update(status="board_verified", board_validation={
        "run_id": metrics["run_id"], "summary": "results/summary.json",
        "scope": metrics["scope"], "formal_accuracy_evaluated": False})
    manifest["files"] = [{"name": path.relative_to(output).as_posix(),
                          "bytes": path.stat().st_size, "sha256": sha256(path)}
                         for path in sorted(output.rglob("*"))
                         if path.is_file() and path != output / "manifest.json"]
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"[板端证据已归档] {output}", flush=True)


def main():
    """解析编译工作目录和新的交付目录,不覆盖已有包."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill", type=int, default=128)
    parser.add_argument("--quantized-prefix", type=Path)
    parser.add_argument("--board-results", type=Path)
    args = parser.parse_args()
    if args.board_results:
        record_board_results(args.work, args.output, args.board_results)
    else:
        package(args.work, args.output, args.prefill, args.quantized_prefix)


if __name__ == "__main__":
    main()
