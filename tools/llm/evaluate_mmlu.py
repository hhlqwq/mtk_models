"""使用固定 MMLU 0-shot 题集对比官方权重的 FP32 参考与板端 Accuracy."""


import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

from tools.llm.artifacts import dla_relative_path
from tools.files import sha256

from tools.evaluation import (
    is_test, record_selection, result_status, scope_metadata,
)
from tools.evaluation import accuracy_result


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


def load_source_tokenizer(source):
    """在内存中兼容官方 BPE merges 的数组格式,保持原始文件和分词规则不变."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    source = Path(source)
    data = json.loads((source / "tokenizer.json").read_text(encoding="utf-8"))
    merges = data.get("model", {}).get("merges", [])
    if not merges or not isinstance(merges[0], list):
        return AutoTokenizer.from_pretrained(source, local_files_only=True)
    if data["model"]["type"] != "BPE":
        raise ValueError("仅为 BPE 分词器兼容 merges 数组格式.")
    for pair in merges:
        if (not isinstance(pair, list) or len(pair) != 2
                or not all(isinstance(part, str) for part in pair)):
            raise ValueError("官方 BPE merges 必须是由两个字符串组成的数组.")
    data["model"]["merges"] = [" ".join(pair) for pair in merges]
    backend = Tokenizer.from_str(json.dumps(data, ensure_ascii=False))
    print("[Tokenizer 兼容] 在内存中转换官方 BPE merges 格式,保留原始资源和 Token 对齐检查.",
          flush=True)
    return AutoTokenizer.from_pretrained(
        source, local_files_only=True, tokenizer_object=backend)


def prepare(args):
    """生成两后端共享的原始题目 Token,严格拒绝截断或丢题."""
    # 仅当前模型或后端需要此依赖,其他评测模式和板端环境不要求安装.
    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer

    if sha256(args.dataset) != DATA_SHA256:
        raise ValueError("MMLU 官方固定测试集哈希不符.")
    rows = parquet.read_table(args.dataset).to_pylist()
    if len(rows) != 14042 or len({row["subject"] for row in rows}) != 57:
        raise ValueError("完整 MMLU 题数或学科数不符.")
    test_questions = getattr(args, "test_questions", 100)
    if is_test() and not 57 <= test_questions <= len(rows):
        raise ValueError("MMLU 子集至少需要 57 题以覆盖全部学科.")
    selected = (select_questions(rows, test_questions, args.seed)
                if is_test() else select_questions(rows, args.limit, args.seed))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    source_tokenizer = (load_source_tokenizer(args.source_model)
                        if getattr(args, "source_model", None) else None)
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
        if source_tokenizer is not None and tokens != source_tokenizer.encode(
                prompt, add_special_tokens=False):
            raise ValueError(f"原始与转换副本 Tokenizer 的题目输入不一致: {source_index}.")
        for letter, choice in zip("ABCD", choices):
            if tokenizer.encode(prompt + " " + letter, add_special_tokens=False) != tokens + choice:
                raise ValueError(f"选项不是完整 Prompt 后的单 Token 延续: {source_index},{letter}.")
            if source_tokenizer is not None and source_tokenizer.encode(
                    prompt + " " + letter, add_special_tokens=False) != tokens + choice:
                raise ValueError(f"原始 Tokenizer 的选项延续不一致: {source_index},{letter}.")
        cases.append({"id": index, "subject": row["subject"],
                      "source_question_id": source_index,
                      "input_ids": tokens, "answer": int(row["answer"])})
        if (index + 1) % 100 == 0 or index + 1 == len(selected):
            print(f"[MMLU Token 核对] {index + 1}/{len(selected)}.", flush=True)
    subject_count = len({case["subject"] for case in cases})
    maximum = max(len(case["input_ids"]) for case in cases)
    print(f"[MMLU 输入] {len(cases)} 题,{subject_count} 学科,0-shot,最长 {maximum} Token.",
          flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"dataset": "cais/mmlu", "dataset_sha256": DATA_SHA256,
                "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
                "split": "test", "shots": 0, "subjects": subject_count,
                "scope": "smoke_subset" if is_test() else (
                    "first_round_stratified_subset" if args.limit else "full_test"),
                "dataset_questions": len(rows), "sampling_seed": args.seed,
                "sampling": "fixed_seed_stratified_subset" if is_test() else (
                    "all_test_questions" if args.limit == 0 else "proportional_subject_largest_remainder"),
                "selected_source_question_ids": selected,
                "questions": len(cases), "maximum_prompt_tokens": maximum,
                "context": args.context, "choice_token_ids": [x[0] for x in choices],
                "chat_template": False, "add_special_tokens": False,
                "scoring": "argmax_single_token_continuation_space_ABCD",
                "choice_continuation_checked": True,
                "source_tokenizer_parity": source_tokenizer is not None,
                "tokenizer_sha256": sha256(args.model / "tokenizer.json"),
                "model_config_sha256": sha256(args.model / "config.json")}
    if is_test():
        record_selection(row["source_question_id"] for row in cases)
    protocol.update(status=result_status(), **scope_metadata())
    # 长度检查结果仍落盘,便于确定实际所需上下文;不输出截断题目.
    (args.output / "length_report.json").write_text(
        json.dumps(protocol, indent=2) + "\n")
    if maximum > args.context:
        raise ValueError(f"完整测试需要至少 {maximum} Token 上下文,"
                         f"当前为 {args.context},不得丢题或截断.")
    (args.output / "cases.json").write_text(json.dumps(cases) + "\n")
    protocol["cases_sha256"] = sha256(args.output / "cases.json")
    if args.graph:
        # 仅当前模型或后端需要此依赖,其他评测模式和板端环境不要求安装.
        from tools.llm.evaluate_quality import export_board_inputs
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
    """将官方存储精度的权重严格加载到 FP32 模型,记录两种精度口径."""
    # 仅当前模型或后端需要此依赖,其他评测模式和板端环境不要求安装.
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
    storage_dtypes = sorted({str(value.dtype) for value in state.values()})
    model.load_state_dict(state, strict=True)
    del state
    model.reference_metadata = {
        "checkpoint_dtypes": storage_dtypes,
        "parameter_dtypes": sorted({str(value.dtype) for value in model.parameters()}),
        "checkpoint_bytes": sum((model_path / name).stat().st_size for name in files),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "origin": "local_reference_from_official_weights_not_official_benchmark",
    }
    return model.eval()


class QuantizedExecutor:
    """用官方软件执行器复核多块 Prefill 与 KV,接口与硬件一致."""

    def __init__(self, graph, contract, *, simulate_fp16=False):
        """加载同一静态图并分配原始 INT16 缓冲区."""
        # 仅当前模型或后端需要此依赖,其他评测模式和板端环境不要求安装.
        from mtk_converter.python.tools.tflite_executor import TFLiteExecutor

        self.executor = TFLiteExecutor(str(graph), simulate_fp16=simulate_fp16,
                                       cache_dequantized_weight=True)
        self.contract = contract
        self.inputs = [np.full(item["shape"], item["zero_point"], dtype="<i2")
                       for item in contract["inputs"]]

    def run(self):
        """返回全部原始输出,避免重复量化或丢失 KV."""
        # 仅当前模型或后端需要此依赖,其他评测模式和板端环境不要求安装.
        from tools.llm.compare_logits import raw_output_list

        feed = {item["name"]: value for item, value in
                zip(self.contract["inputs"], self.inputs)}
        output = self.executor.run(feed, quantize_input=False,
                                   dequantize_output=False)
        return raw_output_list(output, self.contract)

    def close(self):
        """释放软件执行器缓存,不影响其他评价进程."""
        self.executor = None


def evaluate_sdk(args):
    """借用 SDK 浮点驱动采集小批量 MMLU 分数,不发布填充文本的 PPL."""
    import torch
    from types import SimpleNamespace
    from mtk_llm_sdk import benchmark

    if not is_test():
        raise ValueError("SDK 浮点对照仅用于小批量诊断.")
    protocol = json.loads((args.data / "protocol.json").read_text())
    if sha256(args.data / "cases.json") != protocol["cases_sha256"]:
        raise ValueError("SDK 对照题目指纹不符.")
    if sha256(args.model / "config.json") != protocol["model_config_sha256"]:
        raise ValueError("SDK 对照模型配置与输入不符.")
    cases = json.loads((args.data / "cases.json").read_text())
    if not 1 <= len(cases) <= 16:
        raise ValueError("SDK 首轮对照仅允许最多 16 题.")
    context = protocol["context"]
    vocab_size = json.loads((args.model / "config.json").read_text())["vocab_size"]
    ids = []
    for case in cases:
        if not 1 <= len(case["input_ids"]) <= context:
            raise ValueError("SDK 对照输入超出上下文.")
        # 仅右侧补齐,读取真实末位置;填充位置不参与 MMLU 评分.
        ids.extend(case["input_ids"] + [1] * (context - len(case["input_ids"])))
    record_selection(case["source_question_id"] for case in cases)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "questions.jsonl"
    if output.exists():
        raise ValueError("SDK 对照输出已存在,请使用新的日期临时目录.")
    rows, parameter_dtypes = [], set()

    def capture(module, inputs, outputs):
        """只保存四个选项的分数,不落盘完整词表 logits."""
        values = outputs if isinstance(outputs, (tuple, list)) else [outputs]
        logits = next(value for value in values if torch.is_tensor(value)
                      and value.ndim == 3 and value.shape[-1] == vocab_size)
        case = cases[len(rows)]
        scores = logits[0, len(case["input_ids"]) - 1,
                        protocol["choice_token_ids"]].detach().float().cpu().tolist()
        row = {key: case[key] for key in ("id", "subject", "source_question_id", "answer")}
        row.update(scores=scores, prediction=int(np.argmax(scores)))
        with output.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        rows.append(row)
        print(f"[SDK 浮点 MMLU] {len(rows)}/{len(cases)}.", flush=True)

    original_evaluate, original_dataset = benchmark.evaluate_ppl, benchmark.datautils.get_dataset
    original_argv = sys.argv

    def forward(*values, **kwargs):
        """保留 SDK 自身的掩码与前向实现,截取末层输出."""
        models = values[4]
        parameter_dtypes.update(str(parameter.dtype) for model in models
                                for parameter in model.parameters())
        handle = models[-1].register_forward_hook(capture)
        try:
            return original_evaluate(*values, **kwargs)
        finally:
            handle.remove()

    try:
        benchmark.datautils.get_dataset = lambda *values, **kwargs: SimpleNamespace(
            input_ids=torch.tensor([ids]))
        benchmark.evaluate_ppl = forward
        sys.argv = ["sdk_mmlu", str(args.model / "config.json"),
                    "pytorch", "ppl", "--dtype", "float32"]
        try:
            benchmark.main()
        except ModuleNotFoundError as error:
            if error.name != "pytablewriter" or len(rows) != len(cases):
                raise
    finally:
        benchmark.evaluate_ppl, benchmark.datautils.get_dataset = original_evaluate, original_dataset
        sys.argv = original_argv
    if len(rows) != len(cases):
        raise ValueError("SDK 对照未完成全部固定样本.")
    correct = sum(row["prediction"] == row["answer"] for row in rows)
    report = dict(protocol, backend="sdk", evaluated_questions=len(rows), correct=correct,
                  accuracy=correct / len(rows), parameter_dtypes=sorted(parameter_dtypes),
                  padding="right_pad_context_read_last_valid_token", **scope_metadata())
    report["status"] = result_status()
    (args.output / "mmlu_accuracy.json").write_text(json.dumps(report, indent=2) + "\n")


def evaluate(args):
    """统一评分完整题集,落盘每题选项分数和可恢复的进度."""
    if args.backend == "sdk":
        return evaluate_sdk(args)
    protocol = json.loads((args.data / "protocol.json").read_text())
    if sha256(args.data / "cases.json") != protocol["cases_sha256"]:
        raise ValueError("MMLU 题目 Token 文件哈希不符.")
    if protocol.get("evaluation_scope") == "subset" and not is_test():
        raise ValueError("当前为全量入口,不能使用小批量题集.")
    if is_test() and protocol.get("evaluation_scope") == "full":
        raise ValueError("当前为小批量入口,请先准备对应的固定子集题集.")
    cases = json.loads((args.data / "cases.json").read_text())
    if args.diagnostic_questions:
        if args.diagnostic_questions > len(cases):
            raise ValueError("诊断题数超过当前固定题集.")
        order = sorted(cases, key=lambda row: len(row["input_ids"]))
        indices = np.linspace(0, len(order) - 1,
                              args.diagnostic_questions, dtype=int)
        cases = sorted((order[index] for index in indices), key=lambda row: row["id"])
    if is_test():
        record_selection(row["source_question_id"] for row in cases)
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {"backend": args.backend, "cases_sha256": protocol["cases_sha256"]}
    if args.backend == "tflite":
        meta["simulate_fp16"] = bool(getattr(args, "simulate_fp16", False))
    if args.diagnostic_questions:
        meta["diagnostic_case_ids"] = [row["id"] for row in cases]
    hardware = None
    if args.backend == "pytorch":
        # 仅当前模型或后端需要此依赖,其他评测模式和板端环境不要求安装.
        import torch

        meta["config_sha256"] = sha256(args.model / "config.json")
        if meta["config_sha256"] != protocol["model_config_sha256"]:
            raise ValueError("FP32 模型配置与题目 Token 准备配置不一致.")
        device = getattr(args, "device", "cpu")
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("MMLU 配置为 CUDA,但 CUDA 不可用;不自动退回 CPU.")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if protocol.get("tokenizer_sha256") and sha256(
                args.model / "tokenizer.json") != protocol["tokenizer_sha256"]:
            raise ValueError("MMLU 参考的 Tokenizer 与题目准备阶段不一致.")
        reference = load_reference(args.model).to(device)
        meta["reference"] = dict(reference.reference_metadata, device=device,
                                 cuda_matmul_tf32=False if device == "cuda" else None,
                                 torch_version=torch.__version__)
        print(f"[MMLU 参考] 存储 {meta['reference']['checkpoint_dtypes']},"
              f"参数 {meta['reference']['parameter_dtypes']},设备 {device},"
              f"待评分 {len(cases)} 题.", flush=True)

        def score(tokens):
            """取完整 Prompt 的最后一行官方浮点 logits."""
            with torch.inference_mode():
                output = reference(torch.tensor([tokens], device=device), use_cache=False).logits[0, -1]
            return output[protocol["choice_token_ids"]].cpu().numpy().tolist()
    else:
        # 仅当前模型或后端需要此依赖,其他评测模式和板端环境不要求安装.
        from tools.llm.evaluate_board_quality import HardwareModel, transfer_cache

        manifest = json.loads((args.package / "manifest.json").read_text())
        if manifest["context"] != protocol["context"]:
            raise ValueError("交付上下文与 MMLU 协议不符.")
        contract = json.loads((args.data / "board_input_contract.json").read_text())
        expected = manifest["graph_contracts"]["prompt"]
        fields = ("name", "shape", "scale", "zero_point")
        if (contract["tflite_sha256"] != expected["tflite_sha256"]
                or any(len(contract[kind]) != len(expected[kind]) or any(
                    any(actual[field] != saved[field] for field in fields)
                    for actual, saved in zip(contract[kind], expected[kind]))
                    for kind in ("inputs", "outputs"))):
            raise ValueError("MMLU 输入契约与部署 DLA 所属图或 I/O 不一致.")
        graph = (args.package / dla_relative_path(manifest, "prompt")
                 if args.backend == "npu" else args.graph)
        embeddings = args.package / "tokenizer/embedding_int16.bin"
        if protocol.get("tokenizer_sha256") and sha256(
                args.package / "tokenizer/tokenizer.json") != protocol["tokenizer_sha256"]:
            raise ValueError("部署 Tokenizer 与 MMLU 题目准备阶段不一致.")
        for path in ([graph, embeddings] if args.backend == "npu" else [embeddings]):
            relative = path.relative_to(args.package).as_posix()
            entries = [x for x in manifest["files"] if x["name"] == relative]
            if len(entries) != 1:
                raise ValueError(f"交付清单缺少或重复评测文件: {relative}")
            entry = entries[0]
            if sha256(path) != entry["sha256"]:
                raise ValueError("实际板端交付文件哈希不符.")
        meta["graph_sha256"] = sha256(graph)
        if args.backend == "tflite" and meta["graph_sha256"] != contract["tflite_sha256"]:
            raise ValueError("量化软件执行图与契约不一致.")
        if sha256(args.data / "mmlu_inputs.npz") != protocol["inputs_sha256"]:
            raise ValueError("多块 Prefill 输入哈希不符.")
        bank = np.load(args.data / "mmlu_inputs.npz")
        embedding = np.memmap(embeddings, dtype="<i2", mode="r").reshape(
            -1, contract["inputs"][0]["shape"][-1])
        print(f"[MMLU {args.backend}] 加载模型,待评分 {len(cases)} 题.", flush=True)
        hardware = (HardwareModel(graph, contract)
                    if args.backend == "npu" else QuantizedExecutor(
                        graph, contract, simulate_fp16=meta["simulate_fp16"]))

        def score(tokens):
            """执行同协议多块 Prefill,读取最后有效位置的选项 logits."""
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
    if [x["id"] for x in done] != [row["id"] for row in cases[:len(done)]]:
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
                if len(done) % min(100, max(1, len(cases) // 4)) == 0 or len(done) == len(cases):
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
                  elapsed_seconds_this_run=time.perf_counter() - started)
    if "reference" in meta:
        report["reference"] = meta["reference"]
    if args.diagnostic_questions:
        report["scope"] = "multiblock_diagnostic_not_first_round_accuracy"
        report["diagnostic_case_ids"] = [row["id"] for row in cases]
    record_selection(row["source_question_id"] for row in cases)
    report.update(status=result_status(), **scope_metadata())
    if args.backend in ("pytorch", "npu"):
        report["evaluation"] = accuracy_result(args.model_name, args.backend,
            {"MMLU Accuracy": report["accuracy"]}, len(done), "question")
    (args.output / "mmlu_accuracy.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(f"[MMLU完成] {correct}/{len(done)},Accuracy={report['accuracy']:.6f}.",
          flush=True)


def main():
    """解析完整测试的准备、官方参考或板端执行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("prepare", "pytorch", "sdk", "tflite", "npu"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-name", required=True, help="部署配置中的模型标识.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--simulate-fp16", action="store_true",
                        help="仅对量化软件执行启用 SDK FP16 计算模拟,用于硬件误差诊断.")
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=1000,
                        help="默认按学科分层抽取 1000 题;0 表示完整测试集,模型入口按约定显式选择.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--diagnostic-questions", type=int, default=0,
                        help="仅用于覆盖不同长度的多块执行诊断,不代替千题成绩.")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.simulate_fp16 and args.backend != "tflite":
        parser.error("--simulate-fp16 仅用于 tflite 软件诊断.")
    if is_test():
        args.output = args.output / "test"
        if args.backend == "prepare":
            args.test_questions = 100
    if not 0 <= args.diagnostic_questions <= 1000:
        parser.error("诊断题数须位于 [0,1000].")
    if args.backend == "prepare":
        prepare(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
