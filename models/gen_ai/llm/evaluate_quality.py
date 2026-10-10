"""统一 LLM 的 WikiText2 原始文本与 FP32/量化 PPL 协议."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as parquet
import torch
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def sha256(path):
    """分块计算语料与报告资源哈希."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_board_inputs(model, graph_path, output):
    """导出 SDK 同协议的掩码、RoPE 和原始 INT16 I/O 契约."""
    from mtk_converter.python.converters.tflite.schema.tflite.Model import Model
    from mtk_converter.python.converters.tflite.schema.tflite.TensorType import TensorType
    from mtk_llm_sdk.utils import generate_utils, utils

    graph = Model.GetRootAsModel(graph_path.read_bytes(), 0).Subgraphs(0)
    contract = {"tflite_sha256": sha256(graph_path), "inputs": [], "outputs": []}
    for kind, length, index in (("inputs", graph.InputsLength, graph.Inputs),
                                ("outputs", graph.OutputsLength, graph.Outputs)):
        for offset in range(length()):
            tensor = graph.Tensors(index(offset))
            q = tensor.Quantization()
            if tensor.Type() != TensorType.INT16 or q.ScaleLength() != 1:
                raise ValueError("板端 PPL 只支持逐张量 INT16 原生接口.")
            contract[kind].append({"name": tensor.Name().decode(),
                                   "shape": tensor.ShapeAsNumpy().tolist(),
                                   "scale": float(q.Scale(0)),
                                   "zero_point": int(q.ZeroPoint(0))})
    config = utils.resolve_model_classes(str(model / "config.json"),
                                         bypass_tokenizer=True)[0]
    prefill = contract["inputs"][0]["shape"][1]
    mask = generate_utils.generate_mask(config.max_position_embeddings, 0,
                                        prefill, prefill, dtype=np.float32)
    position = generate_utils.get_master_rot_emb(config, np.float32)[:, :, :prefill, :]
    arrays = {}
    for name, values in (("mask", mask), ("pos_emb", position)):
        entry = next(item for item in contract["inputs"] if item["name"] == name)
        if list(values.shape) != entry["shape"]:
            raise ValueError(f"SDK 输入形状不符: {name}")
        arrays[name] = np.clip(np.rint(values / entry["scale"] + entry["zero_point"]),
                               -32768, 32767).astype("<i2")
    np.savez(output / "board_inputs.npz", **arrays)
    contract["board_inputs_sha256"] = sha256(output / "board_inputs.npz")
    (output / "board_input_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n")


def evaluate(args):
    """在同一连续 Token 子集与重置上下文协议下计算两个后端的 PPL."""
    import mtk_llm_sdk.benchmark as benchmark

    if args.text_rows < 1 or (args.blocks is not None and args.blocks < 1):
        raise ValueError("评测行数或块数必须为正整数.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    texts = parquet.read_table(args.corpus, columns=["text"])["text"].to_pylist()
    if sha256(args.corpus) != "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91":
        raise ValueError("固定 WikiText2 测试集哈希不符.")
    # 统一模式固定原始文本行,不同模型按官方词表编码,不固定跨模型 Token 数.
    selected = texts if args.blocks is not None else texts[:args.text_rows]
    text = "\n\n".join(selected)
    tokens = tokenizer(text, return_tensors="pt", verbose=False,
                       add_special_tokens=args.blocks is not None).input_ids
    blocks = args.blocks or tokens.numel() // 128
    length = blocks * 128
    if blocks < 1 or tokens.numel() < length:
        raise ValueError("测试语料不足,不得重复填充语料.")
    excluded_tokens = tokens.numel() - length
    tokens = tokens[:, :length].contiguous()
    protocol = {
        "scope": "host_wikitext2_fixed_text_not_full_board_accuracy",
        "protocol": "legacy_token_prefix" if args.blocks else "fixed_text_rows_v1",
        "text_rows": len(selected),
        "selected_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "excluded_tail_tokens": excluded_tokens,
        "corpus_sha256": sha256(args.corpus), "blocks": blocks,
        "block_tokens": 128, "scored_tokens": blocks * 127,
        "reset_context_each_block": True, "chat_template": False,
        "input_ids": tokens[0].tolist(),
        "model_name": args.model.name,
        "reference_device": args.device if args.backend == "pytorch" else None,
        "model_config_sha256": sha256(args.model / "config.json"),
        "native_prepare_sha256": sha256(args.model.parent / "native_prepare.json"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if args.backend == "pytorch":
        if args.device == "cpu":
            torch.set_num_threads(args.cpu_threads)
        print("[质量参考] 加载官方权重 FP32.", flush=True)
        # 官方 Safetensors 没有 format 元数据,旧 Transformers 加载器会报错.
        # 按官方索引直接加载 Tensor,严格匹配全部权重,不重写权重文件.
        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=torch.float32, attn_implementation="eager")
        index_path = args.model / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            weight_files = sorted(set(index["weight_map"].values()))
        else:
            weight_files = ["model.safetensors"]
        state = {}
        for name in weight_files:
            state.update(load_file(args.model / name))
        model.load_state_dict(state, strict=True)
        del state
        model = model.to(args.device).eval()
        total_nll = 0.0
        with torch.inference_mode():
            for index in range(blocks):
                ids = tokens[:, index * 128:(index + 1) * 128].to(args.device)
                logits = model(ids, use_cache=False).logits[:, :-1].float()
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                    reduction="sum")
                total_nll += loss.item()
                print(f"[质量参考] {index + 1}/{blocks}", flush=True)
        ppl = float(torch.exp(torch.tensor(total_nll / protocol["scored_tokens"])))
        preparation = json.loads((args.model.parent / "native_prepare.json").read_text())
        protocol["demo_references"] = []
        for sample in ([] if args.skip_demo else preparation["demos"]):
            print(f"[官方 Greedy 参考] {sample['id']}", flush=True)
            ids = torch.tensor([sample["input_ids"]], device=args.device)
            with torch.inference_mode():
                output = model.generate(
                    ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=768, do_sample=False,
                    eos_token_id=preparation["stop_token_ids"], pad_token_id=1)
            generated = output[0, ids.shape[1]:].tolist()
            protocol["demo_references"].append({
                "id": sample["id"], "generated_tokens": generated,
                "eos_reached": generated[-1] in preparation["stop_token_ids"],
                "text": tokenizer.decode(generated, skip_special_tokens=False),
            })
    else:
        # Shape fixer 完成后会复制 embedding,评测不得提前修改静态图目录.
        embedding = args.tflite / "embedding_int16.bin"
        if not embedding.exists():
            raise ValueError("请等待 Shape 阶段完成,量化评测需要实际 INT16 embedding.")
        protocol["embedding_sha256"] = sha256(embedding)
        graphs = list(args.tflite.glob("*.tflite"))
        if len(graphs) != 1:
            raise ValueError("量化质量检查要求唯一静态 TFLite.")
        protocol["tflite_sha256"] = sha256(graphs[0])
        export_board_inputs(args.model, graphs[0], args.output)
        (args.output / "board_quality_protocol.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
        if args.backend == "prepare_board":
            print(f"[板端输入已准备] {args.output}", flush=True)
            return
        # 仅在当前评测进程注入本地固定语料,不修改 SDK 或它的全局缓存.
        def get_local_dataset(*unused_args, **unused_kwargs):
            """返回与浮点参考完全相同的固定 Token 子集."""
            return SimpleNamespace(input_ids=tokens)

        benchmark.datautils.get_dataset = get_local_dataset
        responses = Path("responses")
        before = set(responses.glob("*.json")) if responses.exists() else set()
        sys.argv = ["mtk_benchmark_llm", str(args.model / "config.json"),
                    "tflite", "ppl", "-d", "wikitext", "-t", str(args.tflite),
                    "--save"]
        try:
            benchmark.main()
        except ModuleNotFoundError as error:
            if error.name != "pytablewriter":
                raise
            # 可选表格显示在 JSON 保存后执行,仍须核对唯一完整结果.
            print("[报告] 可选表格依赖缺失,校验 SDK 已保存的 JSON.", flush=True)
        created = set(responses.glob("*.json")) - before
        if len(created) != 1:
            raise ValueError("官方评测未生成唯一结果文件.")
        raw = json.loads(created.pop().read_text())
        ppl = raw["results"]["wikitext2"]["perplexity"]
        protocol["sdk_result"] = raw
    protocol.update(backend=args.backend, perplexity=ppl)
    target = args.output / f"{args.backend}_quality.json"
    target.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(f"[质量结果] {args.backend} PPL={ppl:.6f}; {target}", flush=True)


def main():
    """解析已准备模型、固定语料和独立报告目录."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("pytorch", "tflite", "prepare_board"), required=True)
    parser.add_argument("--tflite", type=Path)
    parser.add_argument("--text-rows", type=int, default=128)
    parser.add_argument("--blocks", type=int, help="仅复现旧版固定 Token 子集结果.")
    parser.add_argument("--skip-demo", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    if args.backend in ("tflite", "prepare_board") and args.tflite is None:
        parser.error("量化后端需要 --tflite.")
    evaluate(args)


if __name__ == "__main__":
    main()
