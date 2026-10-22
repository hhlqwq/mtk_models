"""整理 DeepSeek W4A16 双 DLA、原生输入资源及独立板端程序."""

import copy
import json
from pathlib import Path
import shutil
import sys

import yaml

from tools.deploy import arguments
from tools.files import copy_public_tools, read_json, sha256, single_file
from tools.llm.native import settings
from tools.llm.package import graph_contract
from tools.storage import REPO_ROOT


def package_model(native):
    """从本次编译产物建立部署包,不混入旧模型、评价输入或报告."""
    preparation = read_json(native.work / "native_prepare.json")
    config = read_json(native.model / "config.json")
    quantization = read_json(native.work / "quantization_settings.json")
    if (preparation["context"] != native.context
            or quantization["quantized_prefix"] != native.prefix
            or any(quantization[key] != "asym4W_sym16A"
                   for key in ("attention", "mlp", "lm_head"))):
        raise ValueError("本次配置与转换产物不一致,不能打包旧模型.")
    output = native.ctx.work / "package"
    destination = native.ctx.output / "models"
    for path in (output, destination):
        if path.is_symlink() or not path.resolve().is_relative_to(native.ctx.output.resolve()):
            raise ValueError("部署包必须位于本模型资源目录,且不能是链接.")
    if output.exists():
        shutil.rmtree(output)
    for folder in (f"{native.context}c", "tokenizer", "scripts/prompts", "board/tools/llm"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    contracts = {}
    for phase, tokens in (("prompt", native.prefill), ("decode", 1)):
        folder = native.quantized.with_name(f"{native.prefix}_{tokens}t{native.context}c")
        graph = single_file(folder, "*.tflite")
        dla = graph.with_suffix(".dla")
        if not dla.is_file() or dla.stat().st_size == 0:
            raise FileNotFoundError(f"缺少已编译的 {phase} DLA.")
        contracts[phase] = dict(graph_contract(graph), tflite_sha256=sha256(graph))
        shutil.copyfile(dla, output / f"{native.context}c/{phase}.dla")
    embedding = native.quantized / "embedding_int16.bin"
    if embedding.stat().st_size != config["hidden_size"] * config["vocab_size"] * 2:
        raise ValueError("INT16 embedding 大小与词表、hidden size 不一致.")
    shutil.copyfile(embedding, output / "tokenizer/embedding_int16.bin")
    tokenizer = read_json(native.model / "tokenizer.json")
    with (output / "tokenizer/vocab.txt").open("w", encoding="utf-8") as stream:
        for token, index in tokenizer["model"]["vocab"].items():
            stream.write(f"{token}\n{index}\n")
    merges = [" ".join(item) if isinstance(item, list) else item
              for item in tokenizer["model"]["merges"]]
    (output / "tokenizer/merges.txt").write_text(
        "#version\n" + "\n".join(merges) + "\n", encoding="utf-8")
    (output / "tokenizer/added_tokens.yaml").write_text(yaml.safe_dump(
        {item["id"]: item["content"] for item in tokenizer["added_tokens"]},
        allow_unicode=True), encoding="utf-8")
    for prompt in (native.work / "prompts").glob("*.txt"):
        shutil.copyfile(prompt, output / "scripts/prompts" / prompt.name)
    # 保留既有 DeepSeek 原生配置,KV 实际维度以导出图接口为准.
    options = {
        "promptTokenBatchSize": native.prefill, "genTokenBatchSize": 1,
        "cacheSize": native.context, "hiddenSize": config["hidden_size"],
        "numHead": config["num_attention_heads"],
        "numLayer": config["num_hidden_layers"], "maxTokenLength": native.context,
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
        "dlaPromptPaths": [f"./{native.context}c/prompt.dla"],
        "dlaGenPaths": [f"./{native.context}c/decode.dla"],
    }
    (output / "scripts/config-yocto.yaml").write_text(yaml.safe_dump(
        {"modelOptions": options, "runtimeOptions": runtime},
        sort_keys=False, allow_unicode=True), encoding="utf-8")
    for name in ("compile.sh", "evaluate.sh"):
        shutil.copyfile(native.ctx.model / "deploy" / name, output / name)
    for name in ("infer.py", "evaluate.py"):
        shutil.copyfile(native.ctx.model / "deploy/board" / name, output / "board" / name)
    shutil.copyfile(REPO_ROOT / "tools/summary.py", output / "board/summary.py")
    library = native.ctx.work / "libneuron_bridge.so"
    with library.open("rb") as stream:
        header = stream.read(20)
    if header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != 183:
        raise ValueError("桥接库必须为 AArch64 ELF.")
    shutil.copyfile(library, output / "board" / library.name)
    copy_public_tools(output / "board/tools", ("files", "evaluation", "board", "process"))
    for name in ("__init__", "evaluate_board_quality", "evaluate_mmlu"):
        shutil.copyfile(REPO_ROOT / "tools/llm" / f"{name}.py",
                        output / "board/tools/llm" / f"{name}.py")
    snapshot = copy.deepcopy(native.ctx.config)
    # 交付快照不包含主机资源根目录、地址及内部工具链路径.
    snapshot["model"] = {"source": "original", "output_dir": "."}
    snapshot["data"]["dataset_dir"] = "."
    snapshot["board"].update(deploy_dir=".", dataset_dir=".")
    snapshot["conversion"]["context"] = native.context
    snapshot["native"].update(
        prefill=native.prefill, calibration_batches=native.batches,
        rotate=bool(native.rotate), mask_value=native.mask,
        weight_optimization=native.optimization,
        weight_opt_config=Path(native.extra).name if native.extra else "",
        calibration_train=Path(native.train).name if native.train else "")
    (output / "config.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(native.model / "LICENSE", output / "LICENSE")
    (output / "README.md").write_text(
        "# DeepSeek-R1-Distill-Qwen-1.5B\n\n"
        "W4A16 双 DLA 部署包.\n\n"
        "`bash compile.sh` 检查少量输入并保存 examples/output 中的双语示例.\n"
        "单独准备 evaluation/full 后执行 `bash evaluate.sh`,正式结果写入 results/summary.json.\n",
        encoding="utf-8")
    manifest = {
        "model": native.ctx.config["name"], "status": "compiled_pending_board",
        "precision": "asym4W_sym16A", "precision_settings": quantization,
        "preparation": preparation, "context": native.context, "prefill": native.prefill,
        "graph_contracts": contracts,
        "files": [{"name": path.relative_to(output).as_posix(),
                   "bytes": path.stat().st_size, "sha256": sha256(path)}
                  for path in sorted(output.rglob("*")) if path.is_file()],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if destination.exists():
        shutil.rmtree(destination)
    output.rename(destination)
    print(f"[打包完成] {destination}", flush=True)
    return destination


if __name__ == "__main__":
    package_model(settings(arguments(__file__), python_executable=sys.executable))
