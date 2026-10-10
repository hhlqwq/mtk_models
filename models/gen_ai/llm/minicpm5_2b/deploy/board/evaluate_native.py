"""运行原生双语样例,记录完整日志、停止 Token、速度和本次内存观测."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def available_memory():
    """读取系统可用内存,不能单独代表模型驱动内存."""
    text = Path("/proc/meminfo").read_text()
    return int(re.search(r"MemAvailable:\s+(\d+)", text).group(1)) / 1024


def response_body(text):
    """提取原生 CLI 的完整回答以比较文本和 Token 输入结果."""
    match = re.search(r"\[Full\s*Response\]\s*\n(.*?)\n(?:Generated Tokens:|\[Latency\])",
                      text, re.DOTALL)
    if match is None:
        raise ValueError("CLI 日志缺少完整回答.")
    return match.group(1).strip()


def run_sample(root, folder, language, token_mode, limit):
    """执行一个固定输入并采样进程峰值 RSS 和系统可用内存."""
    name = f"{language}_demo" if token_mode else f"{language}_text_input"
    log = folder / f"{name}.log"
    suffix = ".tokens.txt" if token_mode else ".txt"
    command = ["llm_cmdline_tool", "scripts/config-yocto.yaml"]
    if token_mode:
        command.append("--read-tokens")
    command += ["-i", f"scripts/prompts/{language}_demo{suffix}", "-m", str(limit - 1)]
    before = available_memory()
    lowest = before
    peak = 0.0
    started = time.perf_counter()
    print(f"[板端] {name}", flush=True)
    with log.open("wb") as stream:
        process = subprocess.Popen(command, cwd=root, stdout=stream,
                                   stderr=subprocess.STDOUT)
        while process.poll() is None:
            lowest = min(lowest, available_memory())
            try:
                status = Path(f"/proc/{process.pid}/status").read_text()
                match = re.search(r"VmHWM:\s+(\d+)", status)
                if match:
                    peak = max(peak, int(match.group(1)) / 1024)
            except FileNotFoundError:
                pass
            time.sleep(0.1)
    elapsed = time.perf_counter() - started
    text = log.read_text(errors="replace")
    if process.returncode != 0:
        raise RuntimeError(f"原生推理失败: {log}")
    row = {"id": f"{language}_demo", "returncode": process.returncode,
           "command": command, "elapsed_seconds": elapsed,
           "peak_process_rss_mib": peak, "memavailable_initial_mib": before,
           "memavailable_minimum_mib": lowest,
           "memavailable_decrease_mib": before - lowest,
           "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
    if token_mode:
        found = re.search(r"Generated Tokens:\s*\{([^}]+)\}", text)
        if found is None:
            raise ValueError("CLI 日志缺少生成 Token.")
        row["generated_tokens"] = [int(value) for value in found.group(1).split(",")]
        row["eos_reached"] = row["generated_tokens"][-1] in (1, 130073)
        row["prefill_seconds"] = float(re.search(
            r"Done analyzing prompt in ([0-9.eE+-]+)s", text).group(1))
        row["native_reported_decode_tokens_per_second"] = float(re.search(
            r"Generative Mode:\s*([0-9.eE+-]+) tok/s", text).group(1))
        if not row["eos_reached"]:
            raise ValueError(f"样例未完成停止 Token: {name}")
    return row, response_body(text)


def main():
    """执行两种输入模式的双语核对,输出原始板端证据."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id) or args.max_new_tokens < 2:
        parser.error("运行编号或生成上限无效.")
    folder = args.root / "results" / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    report = {"run_id": args.run_id,
              "scope": "authored_bilingual_smoke_not_formal_accuracy",
              "samples": [], "text_input_checks": []}
    for language in ("zh", "en"):
        row, token_text = run_sample(args.root, folder, language, True,
                                     args.max_new_tokens)
        report["samples"].append(row)
        check, text = run_sample(args.root, folder, language, False,
                                  args.max_new_tokens)
        check["text_equals_token_input_response"] = text == token_text
        report["text_input_checks"].append(check)
        if text != token_text:
            raise ValueError(f"文本输入与 Token 输入输出不同: {language}")
    (folder / "board_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"[板端证据完成] {folder}", flush=True)


if __name__ == "__main__":
    main()
