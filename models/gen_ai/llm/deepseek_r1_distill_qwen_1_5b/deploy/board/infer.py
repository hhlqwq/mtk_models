"""使用官方 Token 输入生成双语示例,正式评价时采集定长性能与 CLI RSS."""

import json
import math
import os
import re
import subprocess
import time

from tools.board import context
from tools.evaluation import is_test
from tools.files import read_json


def run_sample(ctx, manifest, name, prompt, limit):
    """调用原生 CLI 并显示工具输出,仅正式评价采集性能与进程高水位."""
    count = len(prompt.read_text(encoding="utf-8").split())
    if count < 1 or count + limit > manifest["context"] or limit < 2:
        raise ValueError(f"输入与生成预算超过已编译上下文: {name}.")
    # 原生 CLI 的 -m 不含第一个生成 Token,总预算需减 1.
    command = ["llm_cmdline_tool", "scripts/config-yocto.yaml", "--read-tokens",
               "-i", str(prompt.resolve()), "-m", str(limit - 1)]
    log = ctx.result / f"{name}.log"
    peak = 0.0
    started = time.monotonic()
    next_progress = started + 10
    print(f"[生成] {name}", flush=True)
    # 日志用于解析原生 CLI 的回答和 Token 计数,同时将输出显示到终端.
    with log.open("w", encoding="utf-8") as stream:
        with subprocess.Popen(command, cwd=ctx.root, stdout=stream,
                              stderr=subprocess.STDOUT) as process:
            with log.open(encoding="utf-8", errors="replace") as reader:
                while True:
                    if is_test():
                        finished = process.poll() is not None
                    else:
                        # wait4 返回本次 CLI 退出时的真实高水位,不以定时采样代替峰值.
                        pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                        finished = pid != 0
                        if finished:
                            process.returncode = os.waitstatus_to_exitcode(status)
                            peak = usage.ru_maxrss / 1024
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end="", flush=True)
                    if finished:
                        break
                    if time.monotonic() >= next_progress:
                        print(f"[运行中] {name}", flush=True)
                        next_progress = time.monotonic() + 10
                    time.sleep(0.1)
    if process.returncode != 0:
        raise RuntimeError(f"原生推理失败,退出码={process.returncode}; 日志: {log}.")
    text = log.read_text(encoding="utf-8", errors="replace")
    answer = re.search(
        r"\[Full\s*Response\]\s*\n(.*?)\n(?:Generated Tokens:|\[Latency\])", text, re.DOTALL)
    token_match = re.search(r"Generated Tokens:\s*\{([^}]+)\}", text)
    if answer is None or token_match is None:
        raise ValueError(f"原生 CLI 缺少回答或 Token 输出: {log}.")
    tokens = [int(value.strip()) for value in token_match.group(1).split(",")]
    if not tokens or len(tokens) > limit:
        raise ValueError("原生 CLI 生成数量与预算不一致.")
    row = {"id": name, "generated_tokens": tokens,
           "eos_reached": tokens[-1] in manifest["preparation"]["stop_token_ids"]}
    row["stop_reason"] = "eos" if row["eos_reached"] else "generation_limit"
    if not is_test():
        prefill = re.search(r"Done analyzing prompt in ([0-9.eE+-]+)s", text)
        decode = re.search(r"Generative Mode:\s*([0-9.eE+-]+) tok/s", text)
        if prefill is None or decode is None:
            raise ValueError("原生 CLI 缺少 Prefill/Decode 性能计数.")
        seconds, speed = float(prefill.group(1)), float(decode.group(1))
        if not all(math.isfinite(value) and value > 0 for value in (seconds, speed, peak)):
            raise ValueError("原生 CLI 性能或进程峰值 RSS 无效.")
        row.update(prefill_seconds=seconds, prefill_input_tokens=count,
                   prefill_tokens_per_second=count / seconds,
                   native_reported_decode_tokens_per_second=speed,
                   peak_process_rss_mib=peak,
                   elapsed_seconds=time.monotonic() - started)
    return row, answer.group(1).strip()


def main():
    """编译阶段只保存双语示例,评价阶段运行统一 128 Token 性能协议."""
    ctx = context(__file__, new_run=True)
    manifest = read_json(ctx.root / "manifest.json")
    if (manifest.get("model") != ctx.config["name"]
            or manifest["context"] != ctx.config["conversion"]["context"]):
        raise ValueError("部署清单与模型配置不一致.")
    if not is_test():
        from evaluate import check_resources

        check_resources(ctx.root)
    ctx.result.mkdir(parents=True, exist_ok=False)
    samples = []
    for language in ("zh", "en"):
        prompt = ctx.root / f"scripts/prompts/{language}_demo.tokens.txt"
        limit = 32 if is_test() else ctx.config["conversion"]["max_new_tokens"]
        row, text = run_sample(ctx, manifest, f"{language}_demo", prompt, limit)
        if is_test():
            output = ctx.root / "examples/output"
            output.mkdir(parents=True, exist_ok=True)
            (output / f"{language}_demo.txt").write_text(text + "\n", encoding="utf-8")
        else:
            samples.append(row)
    if is_test():
        print("[通路检查完成] 双语生成片段已保存,未生成正式指标.", flush=True)
        return
    quality = read_json(ctx.root / "evaluation/full/quality/board_quality_protocol.json")
    tokens = quality["input_ids"][:128]
    if len(tokens) != 128:
        raise ValueError("定长性能输入不足 128 Token.")
    prompt = ctx.result / "performance_128.tokens.txt"
    prompt.write_text(" ".join(map(str, tokens)) + "\n", encoding="utf-8")
    # 独立进程预热不计入正式计数;正式速度使用随后一次 CLI 的内部计时.
    run_sample(ctx, manifest, "warmup", prompt, 2)
    performance, unused_text = run_sample(ctx, manifest, "performance", prompt, 128)
    performance["scope"] = "fixed_128_input_max_128_generated_performance"
    (ctx.result / "board_metrics.json").write_text(json.dumps({
        "run_id": ctx.run_id, "samples": samples, "performance": performance,
        "text_input_checks": [], "functional_failures": [],
        "input_mode": "official_tokenizer_token_ids",
        "unavailable_checks": ["原生文本分词与官方 tokenizer 一致性未验证"],
        "warmup_runs": 1, "measured_runs": 1,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[性能采集完成] 正式指标待精度评价完成后统一发布.", flush=True)


if __name__ == "__main__":
    main()
