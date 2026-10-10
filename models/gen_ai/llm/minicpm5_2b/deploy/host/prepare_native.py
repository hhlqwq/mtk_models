"""准备 MiniCPM5 官方 Token 和原生 GAI 校准输入,不修改原始资源."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from transformers import AutoTokenizer


def sha256(path):
    """分块计算资源哈希,避免一次读入完整权重."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source, output, context):
    """校验官方文件并生成独立配置、双语校准输入和板端 Prompt."""
    manifest = json.loads((source / "source_manifest.json").read_text())
    for entry in manifest["files"]:
        path = source / entry["name"]
        print(f"[校验] {path.name}", flush=True)
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"官方文件校验失败: {path}")
    if output.resolve() == source.resolve() or context < 128:
        raise ValueError("输出必须独立于源目录,上下文不得小于 128.")
    output.mkdir(parents=True, exist_ok=True)
    model = output / "MiniCPM5-2B"
    model.mkdir(exist_ok=True)
    for entry in manifest["files"]:
        src = source / entry["name"]
        dst = model / src.name
        if src.suffix == ".safetensors":
            if not dst.exists():
                dst.symlink_to(src.resolve())
            elif dst.resolve() != src.resolve():
                raise ValueError(f"权重目标已存在且来源不同: {dst}")
        else:
            shutil.copyfile(src, dst)
    # 仅转换独立副本的 BPE merge 序列化格式,不改变词表和规则.
    token_data = json.loads((model / "tokenizer.json").read_text())
    token_data["model"]["merges"] = [
        " ".join(item) if isinstance(item, list) else item
        for item in token_data["model"]["merges"]]
    (model / "tokenizer.json").write_text(
        json.dumps(token_data, ensure_ascii=False) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tokenizer.chat_template = (source / "chat_template.jinja").read_text()
    tokenizer.save_pretrained(model)
    stops = json.loads((source / "config.json").read_text())["eos_token_id"]
    if not isinstance(stops, list) or tokenizer.eos_token_id not in stops:
        raise ValueError("官方停止 Token 配置不符.")
    config = json.loads((source / "config.json").read_text())
    # 推理特殊 Token 以官方 tokenizer/generation 配置为准,修正基础配置遗留值.
    config.update(bos_token_id=tokenizer.bos_token_id,
                  eos_token_id=stops,
                  pad_token_id=tokenizer.pad_token_id,
                  tokenizer="pretrained_fast", max_position_embeddings=context)
    (model / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": "{instruction}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    (output / "minicpm5_preformatter.json").write_text(json.dumps({
        "description": "MiniCPM5 官方单轮 Chat Template.",
        "prompt_input": template, "prompt_no_input": template,
        "response_split": "<|im_start|>assistant\n",
    }, ensure_ascii=False, indent=2) + "\n")
    # 自编双语校准样例只验证部署流程,不作为正式质量基准.
    questions = [
        "请用中文简单介绍你自己.",
        "Please introduce yourself briefly in English.",
        "计算 17 加 25,说明计算过程.",
        "What is 17 plus 25? Explain the calculation.",
        "解释人工智能如何帮助人们分析信息.",
        "Explain how artificial intelligence helps people analyze information.",
        "把这句话译成英文: 北京是中国的首都.",
        "Translate into Chinese: Beijing is the capital of China.",
        "为什么边缘设备需要节省内存?",
        "Why should an edge device use memory efficiently?",
        "写出整理文件的三个步骤.",
        "Give three steps for organizing files.",
        "总结: 小模型便于部署,但仍需要验证输出质量和速度.",
        "Summarize: Small models are easier to deploy but need quality and speed checks.",
        "如果输入过长,应该怎样安排上下文与回答长度?",
        "How should we budget input and output tokens for a limited context?",
    ]
    records = []
    prompts = output / "prompts"
    prompts.mkdir(exist_ok=True)
    for question in questions:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if tokens.count(tokenizer.bos_token_id) != 1:
            raise ValueError("Prompt 必须且只能包含一个 BOS.")
        # GAI SDK 2.7.5 的 tokens 字段采用空格分隔字符串.
        records.append({"tokens": " ".join(str(token) for token in tokens)})
    demos = []
    # Demo 与既有 FP16 基线使用相同 Prompt,校准输入保持独立且不变.
    for name, question in (("zh_demo", "请用中文简单介绍你自己。"),
                           ("en_demo", "Briefly introduce yourself in English.")):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        (prompts / f"{name}.txt").write_text(text)
        (prompts / f"{name}.tokens.txt").write_text(
            " ".join(str(token) for token in tokens) + "\n")
        demos.append({"id": name, "prompt": question, "input_ids": tokens})
    (output / "calibration.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records))
    (output / "native_prepare.json").write_text(json.dumps({
        "source": manifest, "context": context,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id, "stop_token_ids": stops,
        "enable_thinking": False,
        "rope_theta": config["rope_theta"],
        "calibration_scope": "authored_bilingual_smoke_not_formal_accuracy",
        "calibration_samples": len(records),
        "demos": demos,
        "calibration_sha256": sha256(output / "calibration.jsonl"),
        "config_sha256": sha256(model / "config.json"),
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"[准备完成] {model},BOS={tokenizer.bos_token_id},EOS={tokenizer.eos_token_id}",
          flush=True)


def main():
    """解析离线源目录、任务目录和上下文参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=1024)
    args = parser.parse_args()
    prepare(args.source, args.output, args.context)


if __name__ == "__main__":
    main()
