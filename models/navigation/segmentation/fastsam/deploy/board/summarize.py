"""读取 FastSAM 板端配置,保留详细报告并汇总指标与模型大小."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from tools.evaluation import add_test_argument, is_test


def main() -> None:
    """调用已有汇总工具,成功后由工具清理本次 work 目录."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config.json")
    parser.add_argument("--run-id", required=True)
    add_test_argument(parser)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("运行编号只能包含字母、数字、下划线和连字符.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reference_path = root / "models" / ("onnx_reference_test.json" if is_test() else "onnx_reference.json")
    if is_test() and not reference_path.exists():
        reference_path = root / "models/onnx_reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    result = root / config["board"]["results_dir"] / args.run_id
    work = result / "work"
    report = result / "report"
    report.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "coco_summary.log"):
        shutil.copy2(work / "report" / name, report / name)
    print(f"[汇总] 保存核心指标与详细报告: {result}", flush=True)
    subprocess.run([sys.executable, str(root / "board/summarize_board_result.py"), "--model", "fastsam",
                    "--work-dir", str(work), "--output", str(result / "summary.json"),
                    "--run-id", args.run_id, "--reference", str(reference["accuracy"]),
                    "--reference-source", reference["source"], "--reference-report", str(reference_path)], check=True)
    subprocess.run([sys.executable, str(root / "board/model_size.py"),
                    "--models-dir", str(root / "models"), "--include", "model_int8.dla",
                    "--summary", str(result / "summary.json")], check=True)


if __name__ == "__main__":
    main()
