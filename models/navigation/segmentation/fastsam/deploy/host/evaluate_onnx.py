"""评测 FastSAM ONNX 全量精度,在 Docker 内执行."""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

from tools.storage import REPO_ROOT, load_model_config


def run_command(command: list, **kwargs) -> None:
    """显示本步骤实际调用的命令,失败时立即停止."""
    arguments = [str(item) for item in command]
    print(f"[执行] {shlex.join(arguments)}", flush=True)
    subprocess.run(arguments, check=True, **kwargs)


def onnx_eval(config: dict, paths: dict, config_path: Path) -> None:
    """在 Docker 评测全量 ONNX 精度,只输出同协议参考指标."""
    reference = paths["work"] / "onnx_reference.json"
    # 先写入临时文件,校验完成后再发布本次报告.
    pending = paths["work"] / "tmp/onnx_reference.pending.json"
    pending.unlink(missing_ok=True)
    run_command([sys.executable, paths["repo"] / "tools/accuracy/evaluate_onnx.py",
                 "--model", "fastsam", "--onnx", paths["output"] / "model_fp32.onnx",
                 "--dataset-root", paths["dataset"],
                 "--output", pending, "--temp-dir", paths["work"] / "tmp"], cwd=paths["work"])
    if not pending.is_file() or pending.stat().st_size == 0:
        raise FileNotFoundError(f"ONNX 评测未生成报告: {pending}.")
    report = json.loads(pending.read_text(encoding="utf-8"))
    accuracy = report.get("accuracy")
    if (report.get("status") != "complete" or report.get("model") != "fastsam"
            or report.get("metric") != "segm AR@100" or report.get("samples") != 5000
            or type(accuracy) not in (int, float) or not math.isfinite(accuracy)
            or not 0 <= accuracy <= 1):
        raise ValueError(f"FastSAM ONNX 全量评测报告不完整: {pending}.")
    pending.replace(reference)
    print(f"[ONNX 精度] 报告路径: {reference.resolve()}", flush=True)


def main() -> None:
    """直接读取 YAML 配置并执行本文件对应的操作."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "config.yaml")
    parser.add_argument("--work-dir", type=Path, help="覆盖默认的模型外部资源 build 目录.")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_model_config(config_path)
    model = Path(__file__).resolve().parents[2]
    work = args.work_dir or Path(config["model"]["output_dir"]) / "build"
    work = work.resolve()
    for name in ("tmp", "cache"):
        (work / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(work / "tmp"),
                      XDG_CACHE_HOME=str(work / "cache"), TORCH_HOME=str(work / "cache/torch"))
    paths = {"model": model, "repo": REPO_ROOT, "work": work,
             "dataset": Path(config["data"]["dataset_dir"]),
             "output": model / config["model"]["output_dir"]}
    print(f"[步骤] {Path(__file__).stem}; 工作目录: {work}", flush=True)
    onnx_eval(config, paths, config_path)


if __name__ == "__main__":
    main()
