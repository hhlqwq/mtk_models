"""用 Python 完成原生 INT16 双 DLA 的 Prefill、KV 转移和贪心生成."""

import json
import time

import numpy as np

from tools.board import context
from tools.evaluation import is_test
from tools.files import read_json, sha256
from tools.llm.evaluate_board_quality import HardwareModel, transfer_cache
from tools.runtime import peak_rss_kib


def stream_text(tokenizer, tokens, printed, *, final=False):
    """增量打印完整词或中文字符,避免输出未完成的 UTF-8 字符."""
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    if not final:
        text = text.rstrip("\ufffd")
        if text and not (text.endswith("\n") or "\u4e00" <= text[-1] <= "\u9fff"):
            text = text[:max(len(printed), text.rfind(" ") + 1)]
    if not text.startswith(printed):
        raise ValueError("分词器解码修改了已显示文本,无法继续增量输出.")
    print(text[len(printed):], end="", flush=True)
    return text


def requantize(values, source, target):
    """按源和目标量化参数转换 INT16 张量,保留原始形状."""
    if source["scale"] == target["scale"] and source["zero_point"] == target["zero_point"]:
        return values
    values = (values.astype(np.float32) - source["zero_point"]) * source["scale"]
    return np.clip(np.rint(values / target["scale"] + target["zero_point"]),
                   -32768, 32767).astype("<i2")


class Generator:
    """复用两个静态图,输入资源在编译打包阶段由同版本 SDK 准备."""

    def __init__(self, root, manifest):
        """核对清单并加载原始 tokenizer、embedding 和模型专属张量契约."""
        from tokenizers import Tokenizer

        self.manifest = manifest
        self.prompt = self.decode = self.bank = None
        files = {row["name"]: row for row in manifest["files"]}
        required = [f"{manifest['context']}c/{phase}.dla" for phase in ("prompt", "decode")]
        required += ["tokenizer/embedding_int16.bin", "tokenizer/tokenizer.json",
                     "tokenizer/generation_inputs.npz"]
        for name in required:
            if name not in files or sha256(root / name, progress=True) != files[name]["sha256"]:
                raise ValueError(f"生成资源与部署清单不符: {name}.")
        self.contracts = manifest["graph_contracts"]
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer/tokenizer.json"))
        width = self.contracts["prompt"]["inputs"][0]["shape"][-1]
        self.embedding = np.memmap(root / "tokenizer/embedding_int16.bin", dtype="<i2", mode="r").reshape(-1, width)
        try:
            self.bank = np.load(root / "tokenizer/generation_inputs.npz")
            self.prompt = HardwareModel(root / required[0], self.contracts["prompt"])
            self.decode = HardwareModel(root / required[1], self.contracts["decode"])
        except Exception:
            self.close()
            raise

    def feed(self, model, contract, phase, position, tokens):
        """填充 embedding、SDK mask 和 RoPE,保留已有 KV 输入."""
        batch = contract["inputs"][0]["shape"][1]
        for item, buffer in zip(contract["inputs"], model.inputs):
            name = item["name"]
            if name == "input_embeds":
                buffer.fill(item["zero_point"])
                buffer[0, :len(tokens)] = requantize(
                    self.embedding[tokens], self.contracts["prompt"]["inputs"][0], item)
            elif name in ("mask", "pos_emb"):
                np.copyto(buffer, self.bank[f"{phase}_{name}"][position // batch])
            elif not name.startswith("past_"):
                raise ValueError(f"未适配的模型输入: {name}.")

    def seed_decode(self, valid):
        """合并末块之前的 KV 与末块有效 Token,不把 Prefill 填充值带入 Decode."""
        prompt, decode = self.contracts["prompt"], self.contracts["decode"]
        past = {item["name"]: (item, values) for item, values in zip(prompt["inputs"], self.prompt.inputs)}
        current = {item["name"]: (item, values) for item, values in zip(prompt["outputs"], self.prompt.outputs)}
        for item, target in zip(decode["inputs"], self.decode.inputs):
            if not item["name"].startswith("past_"):
                continue
            old_spec, old = past[item["name"]]
            new_spec, new = current[item["name"].replace("past_", "curr_", 1)]
            combined = np.concatenate((requantize(old, old_spec, item),
                                       requantize(new[:, :, :valid, :], new_spec, item)), axis=2)
            if combined.shape[:2] != target.shape[:2] or combined.shape[3:] != target.shape[3:]:
                raise ValueError("Prefill 与 Decode KV 契约不匹配.")
            target.fill(item["zero_point"])
            length = min(target.shape[2], combined.shape[2])
            target[:, :, -length:, :] = combined[:, :, -length:, :]

    def generate(self, tokens, limit, *, stream=False):
        """执行 Prefill 和逐 Token 生成,展示可流式输出,测速默认关闭打印."""
        if (not tokens or limit < 1 or len(tokens) + limit > self.manifest["context"]
                or min(tokens) < 0 or max(tokens) >= len(self.embedding)):
            raise ValueError("生成预算、Token ID 或上下文长度无效.")
        for phase, model in (("prompt", self.prompt), ("decode", self.decode)):
            for item, buffer in zip(self.contracts[phase]["inputs"], model.inputs):
                buffer.fill(item["zero_point"])
        prompt_contract = self.contracts["prompt"]
        decode_contract = self.contracts["decode"]
        batch = prompt_contract["inputs"][0]["shape"][1]
        if stream:
            print(f"[Prefill] 正在处理 {len(tokens)} 个输入 Token.", flush=True)
        started, prefill_npu_ms = time.perf_counter(), 0.0
        for offset in range(0, len(tokens), batch):
            chunk = tokens[offset:offset + batch]
            self.feed(self.prompt, prompt_contract, "prompt", offset, chunk)
            outputs = self.prompt.run()
            prefill_npu_ms += self.prompt.last_npu_ms
            if offset + batch < len(tokens):
                transfer_cache(prompt_contract, outputs, prompt_contract, self.prompt.inputs, append=True)
        prefill_seconds = time.perf_counter() - started
        next_token = int(np.argmax(outputs[0][0, len(chunk) - 1]))
        self.seed_decode(len(chunk))
        generated, decode_npu_ms, decode_calls = [], 0.0, 0
        printed = ""
        if stream:
            print("[输出]", flush=True)
        decode_started = time.perf_counter()
        for step in range(limit):
            generated.append(next_token)
            if stream:
                printed = stream_text(self.tokenizer, generated, printed)
            if next_token in self.manifest["preparation"]["stop_token_ids"] or step + 1 == limit:
                break
            self.feed(self.decode, decode_contract, "decode", len(tokens) + step, [next_token])
            outputs = self.decode.run()
            decode_npu_ms += self.decode.last_npu_ms
            decode_calls += 1
            next_token = int(np.argmax(outputs[0][0, 0]))
            transfer_cache(decode_contract, outputs, decode_contract, self.decode.inputs, append=True)
        decode_seconds = time.perf_counter() - decode_started
        row = {"generated_tokens": generated, "sampling": "greedy_argmax",
               "eos_reached": generated[-1] in self.manifest["preparation"]["stop_token_ids"]}
        row["stop_reason"] = "eos" if row["eos_reached"] else "generation_limit"
        if stream:
            stream_text(self.tokenizer, generated, printed, final=True)
            reason = "遇到停止 Token" if row["eos_reached"] else "达到生成上限"
            print(f"\n[生成结束] {reason},共 {len(generated)} Token.", flush=True)
        if not is_test():
            row.update(prefill_input_tokens=len(tokens), prefill_seconds=prefill_seconds,
                prefill_tokens_per_second=len(tokens) / prefill_seconds,
                decode_tokens_per_second=decode_calls / decode_seconds if decode_calls else None,
                prefill_npu_ms=prefill_npu_ms, decode_npu_ms=decode_npu_ms,
                decode_calls=decode_calls, peak_process_rss_mib=peak_rss_kib() / 1024,
                elapsed_seconds=time.perf_counter() - started)
        return row, self.tokenizer.decode(generated, skip_special_tokens=True)

    def close(self):
        """释放两个 Runtime 和输入资源,不停止其他任务."""
        for model in (self.prompt, self.decode):
            if model is not None:
                model.close()
        if self.bank is not None:
            self.bank.close()


def run(entry):
    """两个模型共享生成流程,语言与停止条件仍由各自部署资源决定."""
    ctx = context(entry, new_run=True)
    manifest = read_json(ctx.root / "manifest.json")
    if manifest["context"] != ctx.config["conversion"]["context"]:
        raise ValueError("配置与已部署 DLA 上下文不一致.")
    if not is_test():
        from evaluate import check_resources

        check_resources(ctx.root)
    ctx.result.mkdir(parents=True, exist_ok=False)
    model = Generator(ctx.root, manifest)
    try:
        samples = []
        languages = ("en",) if ctx.config["name"] == "minicpm5_2b" else ("zh", "en")
        for language in languages:
            path = ctx.root / f"scripts/prompts/{language}_demo.tokens.txt"
            tokens = [int(value) for value in path.read_text().split()]
            prompt = ctx.root / f"scripts/prompts/{language}_demo.txt"
            print(f"[示例] {language}_demo\n[输入,含聊天模板]\n"
                  f"{prompt.read_text(encoding='utf-8')}", flush=True)
            row, text = model.generate(
                tokens, 32 if is_test() else ctx.config["conversion"]["max_new_tokens"],
                stream=True)
            row["id"] = f"{language}_demo"
            if is_test():
                output = ctx.root / "examples/output"
                output.mkdir(parents=True, exist_ok=True)
                name = "sample_1.txt" if ctx.config["name"] == "minicpm5_2b" else f"{language}_demo.txt"
                (output / name).write_text(text + "\n", encoding="utf-8")
                print(f"[示例保存] examples/output/{name}", flush=True)
            else:
                samples.append(row)
        if is_test():
            return
        quality = read_json(ctx.root / "evaluation/full/quality/board_quality_protocol.json")
        tokens = quality["input_ids"][:128]
        if len(tokens) != 128:
            raise ValueError("定长性能输入不足 128 Token.")
        print("[性能] 开始独立预热,关闭文本流式输出.", flush=True)
        model.generate(tokens, 2)
        print("[性能] 开始定长测量,最多生成 128 Token.", flush=True)
        performance, _ = model.generate(tokens, 128)
        print("[性能] 定长测量完成.", flush=True)
        performance["scope"] = "fixed_128_input_max_128_generated_performance"
        (ctx.result / "board_metrics.json").write_text(json.dumps({
            "run_id": ctx.run_id, "samples": samples, "performance": performance,
            "text_input_checks": [], "functional_failures": [],
            "backend": "python_neuron_runtime_hw", "warmup_runs": 1, "measured_runs": 1,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        model.close()
