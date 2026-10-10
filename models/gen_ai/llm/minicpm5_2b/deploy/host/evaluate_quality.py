"""使用固定 WikiText2 子集比较官方 FP32 与量化 TFLite,不冒充板端精度."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pyarrow.parquet as parquet
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def sha256(path):
    """分块计算语料与报告资源哈希."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(args):
    """在同一连续 Token 子集与重置上下文协议下计算两个后端的 PPL."""
    import mtk_llm_sdk.benchmark as benchmark

    if args.blocks < 1:
        raise ValueError("评测块数必须为正整数.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    texts = parquet.read_table(args.corpus, columns=["text"])["text"].to_pylist()
    tokens = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids
    length = args.blocks * 128
    if tokens.numel() < length:
        raise ValueError("测试语料不足,不得重复填充语料.")
    tokens = tokens[:, :length].contiguous()
    protocol = {
        "scope": "host_wikitext2_test_prefix_not_full_board_accuracy",
        "corpus_sha256": sha256(args.corpus), "blocks": args.blocks,
        "block_tokens": 128, "scored_tokens": args.blocks * 127,
        "reset_context_each_block": True, "chat_template": False,
        "input_ids": tokens[0].tolist(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if args.backend == "pytorch":
        print("[质量参考] 加载官方权重 FP32.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.float32,
            attn_implementation="eager").to("cuda").eval()
        total_nll = 0.0
        with torch.inference_mode():
            for index in range(args.blocks):
                ids = tokens[:, index * 128:(index + 1) * 128].to("cuda")
                logits = model(ids, use_cache=False).logits[:, :-1].float()
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                    reduction="sum")
                total_nll += loss.item()
                print(f"[质量参考] {index + 1}/{args.blocks}", flush=True)
        ppl = float(torch.exp(torch.tensor(total_nll / protocol["scored_tokens"])))
    else:
        # 仅在当前评测进程注入本地固定语料,不修改 SDK 或它的全局缓存.
        def get_local_dataset(*unused_args, **unused_kwargs):
            """返回与浮点参考完全相同的固定 Token 子集."""
            return SimpleNamespace(input_ids=tokens)

        benchmark.datautils.get_dataset = get_local_dataset
        responses = Path("responses")
        before = set(responses.glob("*.json")) if responses.exists() else set()
        sys.argv = ["mtk_benchmark_llm", str(args.model / "config.json"),
                    "tflite", "ppl", "-d", "wikitext", "-t", str(args.tflite),
                    "--dtype", "float32", "--save"]
        benchmark.main()
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
    parser.add_argument("--backend", choices=("pytorch", "tflite"), required=True)
    parser.add_argument("--tflite", type=Path)
    parser.add_argument("--blocks", type=int, default=64)
    args = parser.parse_args()
    if args.backend == "tflite" and args.tflite is None:
        parser.error("量化后端需要 --tflite.")
    evaluate(args)


if __name__ == "__main__":
    main()
