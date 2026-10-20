"""将已编译的原生 MiniCPM5 模型整理为 NAS 同类目录结构."""

import json
import os
from pathlib import Path
import re
import shutil

import yaml

from tools.deploy import arguments
from tools.evaluation import is_test
from tools.files import copy_public_tools, sha256, single_file
from tools.llm.native import settings
from tools.llm.package import graph_contract
from tools.storage import REPO_ROOT


def package(work, output, prefill, quantized_prefix=None):
    """复制真实 DLA 与量化资源,生成配置和待板端验证清单."""
    preparation = json.loads((work / "native_prepare.json").read_text())
    context = preparation["context"]
    model = work / "MiniCPM5-2B"
    config = json.loads((model / "config.json").read_text())
    settings_path = work / "quantization_settings.json"
    saved_settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    prefix = saved_settings.get("quantized_prefix", "MiniCPM5-2B_asym4W_sym16A_Overall_hessian")
    base = quantized_prefix or work / "tflite" / prefix
    if base.resolve().parent != (work / "tflite").resolve():
        raise ValueError("量化目录必须位于本次任务的 tflite 目录.")
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
    shutil.copyfile(model / "LICENSE", output / "LICENSE")
    (output / "README.md").write_text(
        f"# MiniCPM5-2B\n\n"
        f"Genio720 原生 {precision},Prefill {prefill},上下文 {context},Decode 1.\n"
        "当前打包状态为 compiled_pending_board,板端结果需另行记录.\n\n"
        "将整个目录复制到板端,执行 `bash run.sh` 全量或 `bash run.sh -test` 快速评价.\n"
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
        "quantized_prefix": base.name,
        "prefill": prefill, "context": context, "files": files,
        "graph_contracts": contracts,
        "toolkit_sha256": "da10e770e2950542ab17c63182b0b932348ab05ffdceb18cbb097555ca6127f5",
    }, ensure_ascii=False, indent=2) + "\n")


def main():
    """按本次模式打包运行入口和同协议输入,拒绝复用旧目录."""
    native = settings(arguments(__file__))
    mode = "test" if is_test() else "full"
    run_id = os.environ["EVAL_RUN_ID"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("运行编号只允许字母、数字、下划线和连字符.")
    evaluation = native.work / "evaluation" / mode
    output = native.ctx.work / "package"
    if (output.is_symlink() or output.resolve().parent != native.ctx.work.resolve()
            or not output.resolve().is_relative_to(native.ctx.output.resolve())):
        raise ValueError("临时部署包必须位于本模型 build/package.")
    if output.exists():
        print("[清理] 替换上一次临时部署包.", flush=True)
        shutil.rmtree(output)
    package(native.work, output, native.prefill, native.quantized)
    shutil.copyfile(native.ctx.model / "deploy/run.sh", output / "run.sh")
    board = output / "board"
    board.mkdir()
    for name in ("infer.py", "evaluate.py", "summarize.py"):
        shutil.copyfile(native.ctx.model / "deploy/board" / name, board / name)
    library = native.ctx.work / "libneuron_bridge.so"
    header = library.read_bytes()[:20]
    if header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != 183:
        raise ValueError("板端桥接库必须为 AArch64 ELF.")
    shutil.copyfile(library, board / library.name)
    copy_public_tools(board / "tools", ("files", "evaluation", "board", "process"))
    shared = board / "tools/llm"
    shared.mkdir()
    for name in ("__init__", "evaluate_native", "evaluate_board_quality", "evaluate_mmlu"):
        shutil.copyfile(REPO_ROOT / "tools/llm" / f"{name}.py", shared / f"{name}.py")
    modes = ["test"] if is_test() else ["full", "test"]
    for prepared in modes:
        shutil.copytree(evaluation.parent / prepared, output / "evaluation" / prepared,
                        ignore=shutil.ignore_patterns("*.npy"))
    quality = json.loads((evaluation / "quality/board_quality_protocol.json").read_text())
    tokens = quality["input_ids"][:128]
    if len(tokens) != 128:
        raise ValueError("统一性能输入不足 128 Token.")
    (output / "scripts/prompts/performance_128.tokens.txt").write_text(
        " ".join(map(str, tokens)) + "\n")
    snapshot = native.ctx.config.copy()
    snapshot["evaluation"] = dict(
        snapshot["evaluation"], prepared_mode=mode, prepared_modes=modes)
    (output / "config.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = [
        {"name": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
         "sha256": sha256(path, progress=True)}
        for path in sorted(output.rglob("*")) if path.is_file() and path != manifest_path]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    destination = native.ctx.output / "models"
    if destination.is_symlink():
        raise ValueError("当前部署包不能是软链接.")
    if destination.exists():
        shutil.rmtree(destination)
    output.rename(destination)
    output = destination
    manifest_path = output / "manifest.json"
    (native.work / "deployment_package.json").write_text(json.dumps({
        "path": str(output), "mode": mode, "run_id": run_id,
        "manifest_sha256": sha256(manifest_path)}, indent=2) + "\n")
    print(f"[打包完成] {output}; 待上传及真实板端验证.", flush=True)


if __name__ == "__main__":
    main()
