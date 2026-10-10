"""将已编译的原生 MiniCPM5 模型整理为 NAS 同类目录结构."""

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import struct

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


def package(work, output, prefill, quantized_prefix=None, quality_results=None,
            bridge_library=None):
    """复制真实 DLA 与量化资源,生成配置和待板端验证清单."""
    preparation = json.loads((work / "native_prepare.json").read_text())
    context = preparation["context"]
    model = work / "MiniCPM5-2B"
    config = json.loads((model / "config.json").read_text())
    base = quantized_prefix or work / "tflite/MiniCPM5-2B_asym4W_sym16A_Overall_hessian"
    if base.resolve().parent != (work / "tflite").resolve():
        raise ValueError("量化目录必须位于本次任务的 tflite 目录.")
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
                          "stopToken": preparation["stop_token_ids"]},
        "tokenizerPath": [f"./tokenizer/{name}" for name in
                          ("merges.txt", "vocab.txt", "added_tokens.yaml")],
        "tokenEmbPath": "./tokenizer/embedding_int16.bin",
        "dlaPromptPaths": [f"./{context}c/prompt.dla"],
        "dlaGenPaths": [f"./{context}c/decode.dla"],
    }
    # 原生 RE2 默认按单个数字切分,改为官方最多三位数字规则.
    # RE2 不支持官方 whitespace lookahead,通用输入优先使用官方 Token.
    runtime["tokenizerRegex"] = (
        r"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
        r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
        r"\s+(?:$|[^\S])|\s+)"
    )
    (output / "scripts/config-yocto.yaml").write_text(
        yaml.safe_dump({"modelOptions": options, "runtimeOptions": runtime},
                       sort_keys=False, allow_unicode=True))
    write_run_script(output)
    quality = None
    if quality_results is not None:
        # 同一模型的两个后端必须使用完全相同的输入,并关联实际转换资源.
        reference = json.loads((quality_results / "pytorch_quality.json").read_text())
        quantized = json.loads((quality_results / "tflite_quality.json").read_text())
        for key in ("protocol", "selected_text_sha256", "corpus_sha256", "input_ids",
                    "blocks", "scored_tokens"):
            if reference[key] != quantized[key]:
                raise ValueError(f"浮点与量化质量协议不一致: {key}")
        if reference["protocol"] != "fixed_text_rows_v1":
            raise ValueError("最终包必须使用统一原始文本评价协议.")
        if quantized["embedding_sha256"] != sha256(embedding) or \
                quantized["tflite_sha256"] != sha256(prompt.with_suffix(".tflite")):
            raise ValueError("质量报告与本次转换资源哈希不符.")
        float_ppl, quant_ppl = reference["perplexity"], quantized["perplexity"]
        if not all(math.isfinite(value) and value > 0 for value in (float_ppl, quant_ppl)):
            raise ValueError("质量报告包含非法 PPL.")
        quality = {"protocol": reference["protocol"], "scope": reference["scope"],
                   "fp32_perplexity": float_ppl, "w4a16_perplexity": quant_ppl,
                   "relative_ppl_increase_percent": (quant_ppl / float_ppl - 1) * 100,
                   "board_npu_accuracy_verified": False}
        folder = output / "results/quality"
        folder.mkdir(parents=True)
        for name in ("pytorch_quality.json", "tflite_quality.json",
                     "board_input_contract.json", "board_inputs.npz"):
            shutil.copyfile(quality_results / name, folder / name)
        shutil.copyfile(quality_results / "board_quality_protocol.json",
                        folder / "board_quality_protocol.json")
        (folder / "summary.json").write_text(
            json.dumps(quality, ensure_ascii=False, indent=2) + "\n")
        shutil.copyfile(Path(__file__).resolve().parents[3] / "evaluate_board_quality.py",
                        output / "scripts/evaluate_quality.py")
        if bridge_library is not None:
            header = bridge_library.read_bytes()[:20]
            if header[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", header, 18)[0] != 183:
                raise ValueError("板端硬件桥接库必须为 AArch64 ELF.")
            shutil.copyfile(bridge_library, output / "scripts/libneuron_bridge.so")
    shutil.copyfile(model / "LICENSE", output / "LICENSE")
    (output / "README.md").write_text(
        f"# MiniCPM5-2B\n\n"
        f"Genio720 原生 W4A16,Prefill {prefill},上下文 {context},Decode 1.\n"
        "当前打包状态为 compiled_pending_board,板端结果需另行记录.\n\n"
        "将整个目录复制到板端,执行 `bash scripts/run.sh`.\n"
        "Demo 使用官方 tokenizer 的 Token 输入,Prompt 文本同时保留供核对.\n"
        f"校准范围: {preparation['calibration_scope']},不代表正式质量基准.\n")
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            print(f"[哈希] {path.relative_to(output)}", flush=True)
            files.append({"name": path.relative_to(output).as_posix(),
                          "bytes": path.stat().st_size, "sha256": sha256(path)})
    (output / "manifest.json").write_text(json.dumps({
        "status": "compiled_pending_board", "preparation": preparation,
        "precision": "asym4W_sym16A", "weight_optimization": "hessian",
        "quantized_prefix": base.name,
        "host_quality": quality,
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
        work / "MiniCPM5-2B", local_files_only=True)
    for row in metrics["samples"]:
        if row["returncode"] != 0 or not row["eos_reached"] or row["generated_tokens"][-1] not in manifest["preparation"]["stop_token_ids"]:
            raise ValueError(f"板端回答未完成: {row['id']}")
        if sha256(results / f"{row['id']}.log") != row["log_sha256"]:
            raise ValueError("原始板端日志哈希不符.")
        row["generated_token_count"] = len(row["generated_tokens"])
        row["text"] = tokenizer.decode(
            row["generated_tokens"], skip_special_tokens=False,
            clean_up_tokenization_spaces=False)
        # 非思考模板已在 Prompt 关闭 think,生成文本无需重复包含结束标签.
        final = tokenizer.decode(row["generated_tokens"], skip_special_tokens=True)
        row["final_answer_present"] = bool(final.strip())
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
    metrics.update(status="board_verified", precision="asym4W_sym16A",
                   prefill=manifest["prefill"], context=manifest["context"],
                   formal_accuracy_evaluated=False,
                   qualitative_observations=["双语样例仅验证推理,正式量化质量须独立评测."],
                   memory_scope="process_vm_hwm_and_sampled_system_memavailable")
    metrics["host_quality"] = manifest.get("host_quality")
    npu_path = results / "npu_quality.json"
    if npu_path.exists():
        npu = json.loads(npu_path.read_text())
        reference = json.loads((output / "results/quality/pytorch_quality.json").read_text())
        for key in ("protocol", "selected_text_sha256", "input_ids", "scored_tokens"):
            if npu[key] != reference[key]:
                raise ValueError(f"板端与浮点质量协议不一致: {key}")
        dla = output / f"{manifest['context']}c/prompt.dla"
        if npu["dla_sha256"] != sha256(dla) or npu["cpu_fallback"]:
            raise ValueError("板端质量报告的 DLA 或硬件路径不符.")
        if len(npu["nll"]) != npu["scored_tokens"] or not all(
                math.isfinite(value) for value in npu["nll"]):
            raise ValueError("板端 NLL 数量或数值非法.")
        metrics["npu_subset_quality"] = {
            "protocol": npu["protocol"], "scope": npu["scope"],
            "perplexity": npu["perplexity"],
            "relative_to_fp32_percent":
                (npu["perplexity"] / reference["perplexity"] - 1) * 100,
            "relative_to_host_tflite_percent":
                (npu["perplexity"] / metrics["host_quality"]["w4a16_perplexity"] - 1) * 100,
            "cpu_fallback": False,
        }
    (output / "results/summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    write_run_script(output)
    readme = (output / "README.md").read_text().replace(
        "当前打包状态为 compiled_pending_board,板端结果需另行记录.",
        "当前状态为 board_verified,仅表示板端推理与双语样例完成,正式质量待评测.")
    readme += "\n默认总生成上限 768 Token.执行 `INPUT_MODE=text bash scripts/run.sh` 可使用原生文本分词.\n"
    readme += "两个固定样例的文本输入结果与官方 Token 输入结果一致.\n\n"
    readme += "| Demo | 生成 Token | Prefill 秒 | Decode Token/s | EOS |\n| --- | ---: | ---: | ---: | --- |\n"
    for row in metrics["samples"]:
        readme += (f"| {row['id']} | {row['generated_token_count']} | "
                   f"{row['prefill_seconds']:.6f} | "
                   f"{row['native_reported_decode_tokens_per_second']:.4f} | 是 |\n")
    readme += "\n原始日志与 Token 均保留,不能将双语 Demo 视为正式精度达标.\n"
    readme += "Prefill 时间不包含加载和模型切换,不作为 TTFT; Runtime 速度按原生 CLI 的 Decode 计数.\n"
    quality = manifest.get("host_quality")
    if quality is not None:
        readme += (f"\n同协议主机测试子集 PPL: FP32 {quality['fp32_perplexity']:.5f},"
                   f"W4A16 {quality['w4a16_perplexity']:.5f},"
                   f"相对变化 {quality['relative_ppl_increase_percent']:.2f}%.\n"
                   "原始质量报告见 `results/quality/`,该结果不代表完整基准或板端 NPU 精度.\n")
    if "npu_subset_quality" in metrics:
        npu = metrics["npu_subset_quality"]
        readme += (f"\n同一子集的实际 MDLA PPL 为 {npu['perplexity']:.5f},"
                   f"相对 FP32 变化 {npu['relative_to_fp32_percent']:.2f}%.\n"
                   "仅为 Prefill 128 图的固定文本子集检查,不是完整 WikiText2 基准.\n")
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
    parser.add_argument("--quality-results", type=Path)
    parser.add_argument("--bridge-library", type=Path)
    parser.add_argument("--board-results", type=Path)
    args = parser.parse_args()
    if args.board_results:
        record_board_results(args.work, args.output, args.board_results)
    else:
        package(args.work, args.output, args.prefill, args.quantized_prefix,
                args.quality_results, args.bridge_library)


if __name__ == "__main__":
    main()
