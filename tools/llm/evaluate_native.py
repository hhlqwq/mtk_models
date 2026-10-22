"""运行原生双语样例,记录完整日志、停止 Token、速度和本次内存观测."""

import argparse
from datetime import date, datetime
import os
from types import SimpleNamespace
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

from tools.evaluation import (
    is_test, result_status, scope_metadata,
)


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


def run_sample(root, folder, language, token_mode, limit, input_path=None,
               require_eos=True):
    """执行一个固定输入并采样进程峰值 RSS 和系统可用内存."""
    name = f"{language}_demo" if token_mode else f"{language}_text_input"
    log = folder / f"{name}.log"
    suffix = ".tokens.txt" if token_mode else ".txt"
    command = ["llm_cmdline_tool", "scripts/config-yocto.yaml"]
    if token_mode:
        command.append("--read-tokens")
    prompt_input = input_path or root / f"scripts/prompts/{language}_demo{suffix}"
    command += ["-i", str(prompt_input.resolve()), "-m", str(limit - 1)]
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
        preparation = json.loads((root / "manifest.json").read_text())["preparation"]
        stop_ids = preparation.get("stop_token_ids", [preparation["eos_token_id"]])
        row["eos_reached"] = row["generated_tokens"][-1] in stop_ids
        row["prefill_seconds"] = float(re.search(
            r"Done analyzing prompt in ([0-9.eE+-]+)s", text).group(1))
        prompt_path = prompt_input
        row["prefill_input_tokens"] = len(re.findall(r"\d+", prompt_path.read_text()))
        if row["prefill_seconds"] <= 0 or row["prefill_input_tokens"] == 0:
            raise ValueError("Prefill 输入数量或耗时无效.")
        row["prefill_tokens_per_second"] = (
            row["prefill_input_tokens"] / row["prefill_seconds"])
        row["native_reported_decode_tokens_per_second"] = float(re.search(
            r"Generative Mode:\s*([0-9.eE+-]+) tok/s", text).group(1))
        if not all(math.isfinite(value) and value > 0 for value in (
                row["prefill_seconds"], row["native_reported_decode_tokens_per_second"])):
            raise ValueError("原生 Prefill 耗时或 Decode 速度无效.")
        if require_eos and not row["eos_reached"]:
            raise ValueError(f"样例未完成停止 Token: {name}")
    return row, response_body(text)


def main():
    """执行两种输入模式的双语核对,输出原始板端证据."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--run-id", default=os.environ.get("EVAL_RUN_ID") or
                        datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--performance-input", type=Path,
                        help="可选固定 128 Token 输入,单独记录统一性能指标.")
    args = parser.parse_args()
    if is_test():
        parser.error("编译示例阶段不允许发布正式报告.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id) or args.max_new_tokens < 2:
        parser.error("运行编号或生成上限无效.")
    temporary = Path("/tmp/hailongcodex") / date.today().isoformat()
    folder = Path(os.environ.get("MTK_RUN_DIR", temporary / ("native_evaluate_" + args.run_id)))
    if folder.is_symlink() or not folder.resolve().is_relative_to(temporary.resolve()):
        parser.error("过程文件必须位于当天日期临时目录.")
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
    if args.performance_input:
        if len(re.findall(r"\d+", args.performance_input.read_text())) != 128:
            raise ValueError("统一性能输入必须恰好为 128 个真实 Token.")
        row, unused_text = run_sample(args.root, folder, "performance", True, 128,
                                      args.performance_input, require_eos=False)
        row.update(unit="token/s", scope="fixed_128_input_max_128_generated_performance",
                   generated_token_count=len(row["generated_tokens"]))
        report["performance"] = row
    report.update(status=result_status(), **scope_metadata())
    (folder / "board_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    from tools.summary import publish

    rows = report["samples"]
    processes = [*rows, *report["text_input_checks"]]
    if "performance" in report:
        processes.append(report["performance"])
    prefill_seconds = sum(row["prefill_seconds"] for row in rows)
    if prefill_seconds <= 0 or any(
            not math.isfinite(row["peak_process_rss_mib"]) or row["peak_process_rss_mib"] <= 0
            for row in processes):
        raise ValueError("原生性能或 RSS 无效.")
    decode_tokens = sum(len(row["generated_tokens"]) for row in rows)
    decode_seconds = sum(len(row["generated_tokens"]) /
                         row["native_reported_decode_tokens_per_second"] for row in rows)
    manifest = json.loads((args.root / "manifest.json").read_text())
    summary = {"schema_version": 1, "run_id": args.run_id, "status": "partial",
               "model": manifest.get("model", args.root.name), "context": manifest["context"],
               "precision": manifest["precision"], "evaluation_scope": "authored_bilingual_generation",
               "accuracy": {"status": "not_implemented", "reason": "尚无正式全量基准精度,生成样例不代替精度."},
               "performance": {
                   "prefill_tokens_per_second": sum(row["prefill_input_tokens"] for row in rows) / prefill_seconds,
                   "decode_tokens_per_second": decode_tokens / decode_seconds,
                   "scope": "中英文固定生成样例,按实际 Token 与原生 CLI 计数汇总",
                   "peak_process_rss_mib": max(row["peak_process_rss_mib"] for row in processes),
                   "memory_scope": "本次生成、文本输入核对及可选测速各进程峰值 RSS 的最大值"}}
    publish(SimpleNamespace(root=args.root, result=folder), summary)
    print(f"[板端证据完成] {folder}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    main()
