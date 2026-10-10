"""准备双语固定样例,比较官方 PyTorch 与自行导出的 ONNX 分片."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def prepare_samples(tokenizer, corpus, max_new_tokens):
    """生成固定 Token 输入,将演示语料与正式评测范围分开标注."""
    rows = [
        {"id": "zh_demo", "prompt": "请用中文简单介绍你自己。"},
        {"id": "en_demo", "prompt": "Briefly introduce yourself in English."},
        {"id": "zh_quality", "text": "北京是中国的首都。人工智能可以帮助人们分析信息。"},
        {"id": "en_quality", "text": "Beijing is the capital of China. Artificial intelligence helps people analyze information."},
    ]
    scope = "authored_bilingual_smoke_not_formal_accuracy"
    if corpus:
        rows.extend(json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()
                    if line.strip())
        scope = "user_supplied_corpus_with_separate_demo"
    samples = []
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"样本 ID 重复: {row['id']}")
        seen.add(row["id"])
        if "prompt" in row:
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                add_generation_prompt=True, tokenize=True)
            samples.append({**row, "input_ids": tokens, "target_ids": []})
        else:
            tokens = tokenizer.encode(row["text"], add_special_tokens=False)
            if not tokens:
                raise ValueError(f"评测文本为空: {row['id']}")
            samples.append({**row, "input_ids": [tokenizer.bos_token_id],
                            "target_ids": tokens})
    return {"samples": samples, "max_new_tokens": max_new_tokens,
            "eos_ids": [tokenizer.eos_token_id], "evaluation_scope": scope}


def evaluate_pytorch(source, samples):
    """用官方 FP32 网络计算教师强制 NLL 与 Greedy Demo 参考."""
    model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager").cuda().eval()
    results = []
    with torch.no_grad():
        for sample in samples["samples"]:
            print(f"[PyTorch] {sample['id']}", flush=True)
            inputs = torch.tensor([sample["input_ids"]], device="cuda")
            if sample["target_ids"]:
                sequence = sample["input_ids"] + sample["target_ids"]
                logits = model(torch.tensor([sequence[:-1]], device="cuda")).logits[0]
                logits = logits[len(sample["input_ids"]) - 1:]
                targets = torch.tensor(sample["target_ids"], device="cuda")
                nll = -logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
                results.append({"id": sample["id"], "nll": nll.cpu().tolist(),
                                "argmax": logits.argmax(-1).cpu().tolist()})
            else:
                outputs = model.generate(inputs, max_new_tokens=samples["max_new_tokens"],
                                         attention_mask=torch.ones_like(inputs),
                                         do_sample=False, temperature=1.0, top_p=1.0,
                                         pad_token_id=samples["eos_ids"][0])
                results.append({"id": sample["id"], "tokens": outputs[0, inputs.shape[1]:].cpu().tolist()})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


class OnnxDecoder:
    """按板端同一静态 KV 协议运行 ONNX,验证导出兼容性."""

    def __init__(self, models):
        """加载静态分片,要求可用 CUDA EP."""
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("原始导出评测要求 CUDAExecutionProvider.")
        self.spec = json.loads((models / "export_manifest.json").read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.graphs = []
        for graph in self.spec["graphs"]:
            print(f"[ONNX 加载] {graph['name']}", flush=True)
            session = ort.InferenceSession(str(models / f"{graph['name']}.onnx"),
                                           sess_options=options,
                                           providers=[("CUDAExecutionProvider",
                                                       {"arena_extend_strategy": "kSameAsRequested"})])
            session.disable_fallback()
            self.graphs.append((graph, session))
        self.embedding = np.memmap(models / "embedding_fp16.bin", mode="r", dtype="<f2",
                                   shape=(self.spec["vocab"], self.spec["hidden"]))
        self.reset()

    def reset(self):
        """重置固定长度缓存与绝对 RoPE 位置."""
        spec = self.spec
        self.cache = [np.zeros((2, 1, spec["kv_heads"], spec["context"] - 1,
                               spec["head_dim"]), dtype=np.float32)
                      for _ in range(spec["layers"])]
        self.position = 0

    def step(self, token):
        """以相同的左侧补零协议运行一个 Token."""
        spec = self.spec
        hidden = self.embedding[token].astype(np.float32).reshape(1, 1, -1)
        frequency = spec["rope_theta"] ** (
            -np.arange(0, spec["head_dim"], 2, dtype=np.float32) / spec["head_dim"])
        angle = np.tile(frequency * self.position, 2).reshape(1, 1, 1, -1)
        mask = np.full((1, 1, 1, spec["context"]), -10000, dtype=np.float32)
        mask[..., spec["context"] - 1 - self.position:] = 0
        logits = np.empty(spec["vocab"], dtype=np.float32)
        layer = 0
        for info, session in self.graphs:
            if info["kind"] == "decoder":
                cache = self.cache[layer]
                hidden, key, value = session.run(None, {
                    "hidden": hidden, "past_key": cache[0], "past_value": cache[1],
                    "cosine": np.cos(angle), "sine": np.sin(angle), "mask": mask})
                cache[:, :, :, :-1] = cache[:, :, :, 1:].copy()
                cache[0, :, :, -1:] = key
                cache[1, :, :, -1:] = value
                layer += 1
            else:
                output = session.run(None, {"hidden": hidden})[0]
                logits[info["start"]:info["start"] + info["count"]] = output.reshape(-1)
        self.position += 1
        return logits


def evaluate_onnx(models, samples):
    """计算相同语料的 NLL,保留逐 Token Argmax 供板端比较."""
    decoder = OnnxDecoder(models)
    results = []
    for sample in samples["samples"]:
        if not sample["target_ids"]:
            continue
        print(f"[ONNX] {sample['id']}", flush=True)
        decoder.reset()
        for token in sample["input_ids"]:
            logits = decoder.step(token)
        nll = []
        argmax = []
        for index, token in enumerate(sample["target_ids"]):
            maximum = float(logits.max())
            nll.append(float(np.log(np.exp(logits.astype(np.float64) - maximum).sum())
                             + maximum - logits[token]))
            argmax.append(int(logits.argmax()))
            if index + 1 < len(sample["target_ids"]):
                logits = decoder.step(token)
        results.append({"id": sample["id"], "nll": nll, "argmax": argmax})
    return results


def run(args):
    """准备 Token 化输入与两后端参考,严格检查导出 NLL 差异."""
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    samples = prepare_samples(tokenizer, args.corpus, args.max_new_tokens)
    spec = json.loads((args.models / "export_manifest.json").read_text())
    for sample in samples["samples"]:
        extra = len(sample["target_ids"]) or args.max_new_tokens
        if len(sample["input_ids"]) + extra > spec["context"]:
            raise ValueError(f"样本超过固定上下文: {sample['id']}")
    (args.models / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    reference = {"evaluation_scope": samples["evaluation_scope"],
                 "pytorch": evaluate_pytorch(args.source, samples),
                 "onnx": evaluate_onnx(args.models, samples)}
    official = {row["id"]: row for row in reference["pytorch"]}
    differences = []
    for row in reference["onnx"]:
        differences.extend(np.abs(np.array(row["nll"]) - official[row["id"]]["nll"]).tolist())
    reference["export_max_token_nll_error"] = max(differences)
    (args.models / "reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2))
    if max(differences) > 0.01:
        raise RuntimeError(f"ONNX 与官方模型不一致,最大 Token NLL 差异={max(differences):.6f}")
    print(f"[参考] 导出一致性通过,最大 Token NLL 差异={max(differences):.6f}", flush=True)


def parse_args():
    """解析离线资源与用户可选的正式评测语料."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
