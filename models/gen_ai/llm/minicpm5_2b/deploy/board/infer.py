"""执行原生 CLI 生成及统一定长性能,保留每次运行的原始证据."""

import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

from tools.board import context
from tools.evaluation import is_test, record_selection, result_status, scope_metadata
from tools.files import read_json, sha256


def available_memory():
    """读取系统可用内存,与推理进程峰值 RSS 分开记录."""
    text = Path("/proc/meminfo").read_text()
    return int(re.search(r"MemAvailable:\s+(\d+)", text).group(1)) / 1024


def run_sample(ctx, manifest, language, token_mode, limit, input_path=None):
    """在此直接调用原生 CLI,打印完整命令并记录日志和内存."""
    if not isinstance(limit, int) or limit < 2:
        raise ValueError("总生成上限必须为不小于 2 的整数.")
    name = f"{language}_demo" if token_mode else f"{language}_text_input"
    suffix = ".tokens.txt" if token_mode else ".txt"
    prompt = input_path or ctx.root / f"scripts/prompts/{language}_demo{suffix}"
    command = ["llm_cmdline_tool", "scripts/config-yocto.yaml"]
    if token_mode:
        command.append("--read-tokens")
    # 原生 CLI 的 -m 统计首 Token 后的调用次数,总生成上限需要减 1.
    command.extend(["-i", str(prompt.resolve()), "-m", str(limit - 1)])
    input_tokens = None
    if token_mode:
        input_tokens = len(prompt.read_text().split())
        if input_tokens < 1 or input_tokens + limit > manifest["context"]:
            raise ValueError(f"输入与生成上限超过上下文容量: {name}.")
    log = ctx.result / f"{name}.log"
    print(f"[板端] {name}; 日志: {log}", flush=True)
    print(f"[执行] cd {shlex.quote(str(ctx.root))} && {shlex.join(command)}", flush=True)
    before = available_memory()
    lowest = before
    peak = 0.0
    started = time.monotonic()
    next_progress = started + 10
    with log.open("wb") as stream:
        with subprocess.Popen(command, cwd=ctx.root, stdout=stream,
                              stderr=subprocess.STDOUT) as process:
            while process.poll() is None:
                lowest = min(lowest, available_memory())
                try:
                    status = Path(f"/proc/{process.pid}/status").read_text()
                    match = re.search(r"VmHWM:\s+(\d+)", status)
                    if match:
                        peak = max(peak, int(match.group(1)) / 1024)
                except FileNotFoundError:
                    pass
                now = time.monotonic()
                if now >= next_progress:
                    print(f"[运行中] {name}: {now - started:.0f}s.", flush=True)
                    next_progress = now + 10
                time.sleep(0.1)
    if process.returncode != 0:
        raise RuntimeError(f"原生推理失败,退出码={process.returncode}; 日志: {log}.")
    text = log.read_text(errors="replace")
    answer = re.search(
        r"\[Full\s*Response\]\s*\n(.*?)\n(?:Generated Tokens:|\[Latency\])",
        text, re.DOTALL)
    if answer is None:
        raise ValueError(f"CLI 日志缺少完整回答: {log}.")
    if is_test():
        print(f"[示例完成] {name}.", flush=True)
        return {"id": name}, answer.group(1).strip()
    if peak <= 0:
        raise ValueError(f"未采集到推理进程峰值 RSS: {log}.")
    row = {
        "id": name, "returncode": process.returncode, "command": command,
        "elapsed_seconds": time.monotonic() - started,
        "peak_process_rss_mib": peak, "memavailable_initial_mib": before,
        "memavailable_minimum_mib": lowest,
        "memavailable_decrease_mib": before - lowest, "log_sha256": sha256(log),
    }
    if token_mode:
        tokens = re.search(r"Generated Tokens:\s*\{([^}]+)\}", text)
        prompt_time = re.search(r"Done analyzing prompt in ([0-9.eE+-]+)s", text)
        decode_speed = re.search(r"Generative Mode:\s*([0-9.eE+-]+) tok/s", text)
        if None in (tokens, prompt_time, decode_speed):
            raise ValueError(f"CLI 日志缺少 Token 或性能计数: {log}.")
        generated = [int(value.strip()) for value in tokens.group(1).split(",")]
        if not generated or len(generated) > limit:
            raise ValueError(f"CLI 生成 Token 数量与上限不符: {log}.")
        seconds = float(prompt_time.group(1))
        speed = float(decode_speed.group(1))
        if not all(math.isfinite(value) and value > 0 for value in (seconds, speed)):
            raise ValueError(f"CLI 性能计数无效: {log}.")
        stops = manifest["preparation"]["stop_token_ids"]
        reached_eos = generated[-1] in stops
        row.update(
            generated_tokens=generated, eos_reached=reached_eos,
            stop_reason="eos" if reached_eos else "no_eos_within_limit",
            prefill_seconds=seconds, prefill_input_tokens=input_tokens,
            prefill_tokens_per_second=input_tokens / seconds,
            native_reported_decode_tokens_per_second=speed)
    print(f"[生成完成] {name}: {row['elapsed_seconds']:.2f}s.", flush=True)
    return row, answer.group(1).strip()


def main():
    """快速模式限制生成长度,固定性能协议保持 128 输入与 128 输出上限."""
    ctx = context(__file__)
    manifest = read_json(ctx.root / "manifest.json")
    if ctx.config["conversion"]["context"] != manifest["context"]:
        raise ValueError("配置与已有 DLA 上下文不一致.")
    ctx.result.mkdir(parents=True, exist_ok=False)
    if is_test():
        unused, text = run_sample(ctx, manifest, "en", True, 32)
        output = ctx.root / "examples/output"
        output.mkdir(parents=True, exist_ok=True)
        (output / "sample_1.txt").write_text(text + "\n", encoding="utf-8")
        return
    quality = read_json(ctx.root / "evaluation/full/quality/board_quality_protocol.json")
    tokens = quality["input_ids"][:128]
    if len(tokens) != 128:
        raise ValueError("定长性能输入不足 128 Token.")
    performance_input = ctx.result / "performance_128.tokens.txt"
    performance_input.write_text(" ".join(map(str, tokens)) + "\n")
    limit = 32 if is_test() else ctx.config["conversion"]["max_new_tokens"]
    report = {"run_id": ctx.run_id, "scope": "generation_smoke_and_fixed_performance",
              "samples": [], "text_input_checks": [], "functional_failures": []}
    row, answer = run_sample(ctx, manifest, "en", True, limit)
    check, text = run_sample(ctx, manifest, "en", False, limit)
    check["text_equals_token_input_response"] = answer == text
    if answer != text:
        report["functional_failures"].append("英文文本与 Token 输入输出不同.")
    if not is_test() and not row["eos_reached"]:
        report["functional_failures"].append("英文生成上限内未正常停止.")
    report["samples"].append(row)
    report["text_input_checks"].append(check)
    performance, unused_text = run_sample(
        ctx, manifest, "performance", True, 128,
        performance_input)
    if performance["prefill_input_tokens"] != 128:
        raise ValueError("统一性能输入必须恰好包含 128 个 Token.")
    performance["scope"] = "fixed_128_input_max_128_generated_performance"
    report["performance"] = performance
    record_selection(("en",))
    report.update(status=("generation_checks_failed" if report["functional_failures"]
                          else result_status()), **scope_metadata())
    (ctx.result / "board_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"[生成与性能完成] {ctx.result}", flush=True)
    for failure in report["functional_failures"]:
        print(f"[生成检查未通过] {failure} 将继续执行独立精度评价.", flush=True)


if __name__ == "__main__":
    main()
