"""使用固定 MMLU 0-shot 题集对比官方 FP32 与板端 NPU Accuracy."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np


DATA_SHA256 = "74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7"


def select_questions(rows, limit, seed):
    """按学科题数比例分层采样,固定清单且不读取答案进行选择."""
    if limit == 0:
        return list(range(len(rows)))
    if not 57 <= limit <= len(rows):
        raise ValueError("首轮题数须覆盖 57 学科且不超过完整题数.")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["subject"]].append(index)
    quotas = {name: max(1, int(limit * len(indices) / len(rows)))
              for name, indices in groups.items()}
    order = sorted(groups, key=lambda name: (
        -(limit * len(groups[name]) / len(rows) - quotas[name]), name))
    if sum(quotas.values()) > limit:
        raise ValueError("题数过少,无法按该学科比例分配.")
    for name in order[:limit - sum(quotas.values())]:
        quotas[name] += 1
    generator = random.Random(seed)
    selected = []
    for name in sorted(groups):
        selected.extend(generator.sample(groups[name], quotas[name]))
    if len(selected) != limit:
        raise ValueError("分层抽样题数不符.")
    return sorted(selected)


def sha256(path):
    """分块计算数据、协议和交付资源哈希."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(args):
    """生成两后端共享的原始题目 Token,严格拒绝截断或丢题."""
    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer

    if sha256(args.dataset) != DATA_SHA256:
        raise ValueError("MMLU 官方固定测试集哈希不符.")
    rows = parquet.read_table(args.dataset).to_pylist()
    if len(rows) != 14042 or len({row["subject"] for row in rows}) != 57:
        raise ValueError("完整 MMLU 题数或学科数不符.")
    selected = select_questions(rows, args.limit, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    choices = [tokenizer.encode(" " + letter, add_special_tokens=False)
               for letter in "ABCD"]
    if any(len(tokens) != 1 for tokens in choices):
        raise ValueError("选项并非单 Token,不能用最后一行 logits 替代序列评分.")
    cases = []
    for index, source_index in enumerate(selected):
        row = rows[source_index]
        subject = row["subject"].replace("_", " ")
        prompt = ("The following are multiple choice questions (with answers) "
                  f"about {subject}.\n\n{row['question']}\n")
        prompt += "\n".join(f"{letter}. {choice}" for letter, choice in
                            zip("ABCD", row["choices"])) + "\nAnswer:"
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        cases.append({"id": index, "subject": row["subject"],
                      "source_question_id": source_index,
                      "input_ids": tokens, "answer": int(row["answer"])})
    maximum = max(len(case["input_ids"]) for case in cases)
    print(f"[MMLU 输入] {len(cases)} 题,57 学科,0-shot,最长 {maximum} Token.",
          flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"dataset": "cais/mmlu", "dataset_sha256": DATA_SHA256,
                "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
                "split": "test", "shots": 0, "subjects": 57,
                "scope": "first_round_stratified_subset" if args.limit else "full_test",
                "dataset_questions": len(rows), "sampling_seed": args.seed,
                "sampling": "proportional_subject_largest_remainder",
                "selected_source_question_ids": selected,
                "questions": len(cases), "maximum_prompt_tokens": maximum,
                "context": args.context, "choice_token_ids": [x[0] for x in choices],
                "chat_template": False, "add_special_tokens": False,
                "scoring": "argmax_single_token_continuation_space_ABCD",
                "model_config_sha256": sha256(args.model / "config.json")}
    # 长度检查结果仍落盘,便于确定实际所需上下文;不输出截断题目.
    (args.output / "length_report.json").write_text(
        json.dumps(protocol, indent=2) + "\n")
    if maximum > args.context:
        raise ValueError(f"完整测试需要至少 {maximum} Token 上下文,"
                         f"当前为 {args.context},不得丢题或截断.")
    (args.output / "cases.json").write_text(json.dumps(cases) + "\n")
    protocol["cases_sha256"] = sha256(args.output / "cases.json")
    if args.graph:
        from evaluate_quality import export_board_inputs
        from mtk_llm_sdk.utils import generate_utils, utils

        export_board_inputs(args.model, args.graph, args.output)
        contract = json.loads((args.output / "board_input_contract.json").read_text())
        config = utils.resolve_model_classes(str(args.model / "config.json"),
                                             bypass_tokenizer=True)[0]
        if config.max_position_embeddings != args.context:
            raise ValueError("模型配置与 MMLU 上下文不一致.")
        master = generate_utils.get_master_rot_emb(config, np.float32)
        masks, positions = [], []
        for offset in range(0, args.context, 128):
            values = {"mask": generate_utils.generate_mask(
                args.context, offset, 128, 128,
                mask_value=config.mask_value, dtype=np.float32),
                "pos_emb": master[:, :, offset:offset + 128, :]}
            for name, bank in (("mask", masks), ("pos_emb", positions)):
                entry = next(x for x in contract["inputs"] if x["name"] == name)
                value = values[name] / entry["scale"] + entry["zero_point"]
                if values[name].shape != tuple(entry["shape"]) or np.any(
                        value < -32768.5) or np.any(value > 32767.5):
                    raise ValueError(f"SDK 多块输入形状或量化范围不符: {name}")
                bank.append(np.rint(value).astype("<i2"))
        np.savez(args.output / "mmlu_inputs.npz", mask=np.stack(masks),
                 pos_emb=np.stack(positions))
        protocol["inputs_sha256"] = sha256(args.output / "mmlu_inputs.npz")
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n")


def load_reference(model_path):
    """按官方 Safetensors 严格加载 FP32 权重,不重写官方资源."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.set_num_threads(8)
    model = AutoModelForCausalLM.from_config(
        AutoConfig.from_pretrained(model_path, local_files_only=True),
        torch_dtype=torch.float32, attn_implementation="eager")
    index = model_path / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) \
        if index.exists() else ["model.safetensors"]
    state = {}
    for name in files:
        print(f"[MMLU FP32] 加载 {name}.", flush=True)
        state.update(load_file(model_path / name, device="cpu"))
    model.load_state_dict(state, strict=True)
    return model.eval()


def evaluate(args):
    """统一评分完整题集,落盘每题选项分数和可恢复的进度."""
    protocol = json.loads((args.data / "protocol.json").read_text())
    if sha256(args.data / "cases.json") != protocol["cases_sha256"]:
        raise ValueError("MMLU 题目 Token 文件哈希不符.")
    cases = json.loads((args.data / "cases.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {"backend": args.backend, "cases_sha256": protocol["cases_sha256"]}
    hardware = None
    if args.backend == "pytorch":
        import torch

        meta["config_sha256"] = sha256(args.model / "config.json")
        if meta["config_sha256"] != protocol["model_config_sha256"]:
            raise ValueError("FP32 模型配置与题目 Token 准备配置不一致.")
        reference = load_reference(args.model)

        def score(tokens):
            """取完整 Prompt 的最后一行官方浮点 logits."""
            with torch.inference_mode():
                output = reference(torch.tensor([tokens])).logits[0, -1]
            return output[protocol["choice_token_ids"]].numpy().tolist()
    else:
        from evaluate_board_quality import HardwareModel, transfer_cache

        manifest = json.loads((args.package / "manifest.json").read_text())
        if manifest["context"] != protocol["context"]:
            raise ValueError("交付上下文与 MMLU 协议不符.")
        contract = json.loads((args.data / "board_input_contract.json").read_text())
        graph = args.package / f"{manifest['context']}c/prefill.dla"
        embeddings = args.package / "tokenizer/embedding_int16.bin"
        for path in (graph, embeddings):
            entry = next(x for x in manifest["files"] if x["name"] ==
                         path.relative_to(args.package).as_posix())
            if sha256(path) != entry["sha256"]:
                raise ValueError("实际板端交付文件哈希不符.")
        meta["dla_sha256"] = sha256(graph)
        if sha256(args.data / "mmlu_inputs.npz") != protocol["inputs_sha256"]:
            raise ValueError("多块 Prefill 输入哈希不符.")
        bank = np.load(args.data / "mmlu_inputs.npz")
        embedding = np.memmap(embeddings, dtype="<i2", mode="r").reshape(
            -1, contract["inputs"][0]["shape"][-1])
        hardware = HardwareModel(args.library, graph, contract)

        def score(tokens):
            """执行多块真实 NPU Prefill,读取最后有效位置的选项 logits."""
            for entry, buffer in zip(contract["inputs"], hardware.inputs):
                buffer.fill(entry["zero_point"])
            for offset in range(0, len(tokens), 128):
                chunk = tokens[offset:offset + 128]
                hardware.inputs[0].fill(0)
                hardware.inputs[0][0, :len(chunk)] = embedding[chunk]
                for index, item in enumerate(contract["inputs"]):
                    if item["name"] in ("mask", "pos_emb"):
                        np.copyto(hardware.inputs[index],
                                  bank[item["name"]][offset // 128])
                outputs = hardware.run()
                if offset + 128 < len(tokens):
                    transfer_cache(contract, outputs, contract,
                                   hardware.inputs, append=True)
            spec = contract["outputs"][0]
            raw = outputs[0][0, len(chunk) - 1, protocol["choice_token_ids"]]
            return ((raw.astype(np.float32) - spec["zero_point"]) *
                    spec["scale"]).tolist()
    meta_path = args.output / "run_manifest.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError("断点结果对应不同模型或题集,请使用独立输出目录.")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    results = args.output / "questions.jsonl"
    done = [json.loads(line) for line in results.read_text().splitlines()] \
        if results.exists() else []
    if [x["id"] for x in done] != list(range(len(done))):
        raise ValueError("断点题目顺序或数量不符.")
    started = time.perf_counter()
    try:
        with results.open("a") as stream:
            for case in cases[len(done):]:
                values = score(case["input_ids"])
                row = {"id": case["id"], "subject": case["subject"],
                       "source_question_id": case["source_question_id"],
                       "answer": case["answer"], "scores": values,
                       "prediction": int(np.argmax(values))}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                done.append(row)
                if len(done) % 100 == 0:
                    print(f"[MMLU {args.backend}] {len(done)}/{len(cases)}.",
                          flush=True)
    finally:
        if hardware is not None:
            hardware.close()
    groups = defaultdict(list)
    for row in done:
        groups[row["subject"]].append(row["prediction"] == row["answer"])
    correct = sum(sum(values) for values in groups.values())
    report = dict(protocol, backend=args.backend, correct=correct,
                  evaluated_questions=len(done), accuracy=correct / len(done),
                  subject_accuracy={name: sum(values) / len(values)
                                    for name, values in groups.items()},
                  elapsed_seconds_this_run=time.perf_counter() - started,
                  quality_qualified=False)
    (args.output / "mmlu_accuracy.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(f"[MMLU完成] {correct}/{len(done)},Accuracy={report['accuracy']:.6f}.",
          flush=True)


def main():
    """解析完整测试的准备、官方参考或板端执行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("prepare", "pytorch", "npu"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=1000,
                        help="首轮默认 1000 题,0 表示完整测试集.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "prepare":
        prepare(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
