#!/usr/bin/env python3
"""合并 AISHELL-1 与 LibriSpeech 正式评测结果."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析联合报告命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aishell-summary", type=Path, required=True)
    parser.add_argument("--aishell-run-id", required=True)
    parser.add_argument("--librispeech-summary", type=Path, required=True)
    parser.add_argument("--librispeech-run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_complete_summary(path: Path, expected_dataset: str) -> dict:
    """读取并验证单数据集正式汇总.

    Args:
        path: 单数据集 summary.json 路径.
        expected_dataset: 预期的数据集标识.

    Returns:
        通过完整性检查的汇总字典.

    Raises:
        RuntimeError: 汇总不存在、数据集不匹配或运行不完整.
    """
    if not path.is_file():
        raise RuntimeError(f"缺少正式汇总: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("dataset") != expected_dataset:
        raise RuntimeError(
            f"汇总数据集不匹配: {path}, "
            f"期望 {expected_dataset}, 实际 {summary.get('dataset')}")
    expected_metric = "cer" if expected_dataset == "aishell1" else "wer"
    complete = (
        summary.get("status") == "complete"
        and summary.get("successful_samples") == summary.get("expected_samples")
        and summary.get("failed_samples") == 0
        and summary.get("missing_samples") == 0
        and summary.get("accuracy", {}).get("metric") == expected_metric
        and summary.get("framework_accuracy", {}).get("metric")
        == expected_metric
        and summary.get("framework_parity") is not None
        and summary.get("performance", {}).get("overall") is not None
    )
    if not complete:
        raise RuntimeError(f"正式汇总不完整: {path}")
    return summary


def file_sha256(path: Path) -> str:
    """计算文件 SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_run(summary: dict, run_id: str, summary_path: Path) -> dict:
    """提取联合报告所需的单数据集字段."""
    return {
        "run_id": run_id,
        "status": summary["status"],
        "expected_samples": summary["expected_samples"],
        "successful_samples": summary["successful_samples"],
        "failed_samples": summary["failed_samples"],
        "missing_samples": summary["missing_samples"],
        "truncated_samples": summary.get("truncated_samples", 0),
        "accuracy": summary["accuracy"],
        "framework_accuracy": summary.get("framework_accuracy"),
        "framework_parity": summary.get("framework_parity"),
        "performance": summary["performance"],
        "host_preprocessing": summary.get("host_preprocessing"),
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
    }


def write_markdown(path: Path, combined: dict) -> None:
    """写入双数据集联合 Markdown 报告."""
    lines = [
        "# Whisper-Tiny 双数据集正式评测结果\n\n",
        f"- 联合状态：`{combined['status']}`\n",
        f"- 生成时间：`{combined['generated_at_utc']}`\n\n",
        "| 数据集 | Run ID | 覆盖 | NPU 指标 | OpenAI 指标 | "
        "NPU 总耗时 Mean/P95（ms） | NPU RTF Mean/P95 |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for dataset, run in combined["datasets"].items():
        accuracy = run["accuracy"]
        framework_accuracy = run["framework_accuracy"]
        performance = run["performance"]["overall"]
        npu_metric = f"{accuracy['metric'].upper()} {accuracy['value']:.6f}"
        if framework_accuracy:
            framework_metric = (
                f"{framework_accuracy['metric'].upper()} "
                f"{framework_accuracy['value']:.6f}")
        else:
            framework_metric = "缺失"
        lines.append(
            f"| {dataset} | `{run['run_id']}` | "
            f"{run['successful_samples']}/{run['expected_samples']} | "
            f"{npu_metric} | {framework_metric} | "
            f"{performance['npu_total_ms']['mean']:.3f}/"
            f"{performance['npu_total_ms']['p95']:.3f} | "
            f"{performance['npu_rtf']['mean']:.6f}/"
            f"{performance['npu_rtf']['p95']:.6f} |\n")
    lines.extend([
        "\n只有两套单数据集汇总都为 `complete`、失败和缺失均为 0 时,"
        "本报告才会生成并标记为 `complete`.\n",
    ])
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    """验证两套结果并生成联合 JSON 与 Markdown 报告."""
    args = parse_args()
    aishell = load_complete_summary(args.aishell_summary, "aishell1")
    librispeech = load_complete_summary(
        args.librispeech_summary, "librispeech")
    combined = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "AISHELL-1 test": extract_run(
                aishell, args.aishell_run_id, args.aishell_summary),
            "LibriSpeech test-clean": extract_run(
                librispeech,
                args.librispeech_run_id,
                args.librispeech_summary),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    write_markdown(args.output_dir / "report.md", combined)
    print(f"[OK] 双数据集联合汇总: {summary_path}")


if __name__ == "__main__":
    main()
