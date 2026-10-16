"""上传 FastSAM 模型、程序与配置,在编译容器内执行."""

import argparse
import json
import math
import os
from pathlib import Path
import re
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


def upload(config: dict, paths: dict, config_path: Path) -> None:
    """准备少量示例并上传部署文件,不在板端启动推理."""
    reference = paths["work"] / "onnx_reference.json"
    if not reference.is_file() or reference.stat().st_size == 0:
        raise FileNotFoundError(
            f"缺少 ONNX 精度报告: {reference}. 请先执行 deploy/host/evaluate_onnx.py.")
    report = json.loads(reference.read_text(encoding="utf-8"))
    accuracy = report.get("accuracy")
    if (report.get("status") != "complete" or report.get("model") != "fastsam"
            or report.get("metric") != "segm AR@100" or report.get("samples") != 5000
            or type(accuracy) not in (int, float) or not math.isfinite(accuracy)
            or not 0 <= accuracy <= 1):
        raise ValueError("缺少完整的 FastSAM 同协议 ONNX 参考精度,请先执行 evaluate_onnx.py.")
    board = config["board"]
    platform_path = paths["repo"] / "configs/platforms/genio_720.yaml"
    platform = yaml.safe_load(platform_path.read_text(encoding="utf-8"))
    board_host = os.environ.get("MTK_BOARD_HOST", platform["board_host"])
    destination = board["deploy_dir"]
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", destination) or any(
            part in ("", ".", "..") for part in destination.split("/")[1:]):
        raise ValueError("board.deploy_dir 必须是无空格、空段或上级引用的绝对目录.")
    ssh_options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    samples = paths["work"] / "samples.json"
    run_command([sys.executable, paths["repo"] / "tools/prepare_examples.py", "--model",
                 "fastsam", "--dataset-root", paths["dataset"],
                 "--output", samples])
    # 板端读取标准 JSON,避免为了配置新增 PyYAML 依赖.
    snapshot = paths["work"] / "config.json"
    snapshot.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    deploy = paths["model"] / "deploy"
    groups = {
        "": [deploy / "run.sh", snapshot],
        "models": [paths["output"] / "model_int8.dla",
                   paths["work"] / "runtime_config.csv", reference],
        "board": [deploy / "board/infer.py", deploy / "board/full_accuracy_board.py",
                  deploy / "board/render.py", deploy / "board/summarize.py",
                  deploy / "board/accuracy_protocol.json", paths["work"] / "fastsam_board",
                  *[paths["repo"] / "tools" / name for name in
                    ("render_examples.py", "summarize_board_result.py", "model_size.py")]],
        "examples/input": [samples,
                           *sorted((paths["model"] / "examples/input").glob("sample_*.jpg"))],
    }
    # 先检查所有本地文件,缺失时不开始上传.
    for files in groups.values():
        for path in files:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"上传文件缺失或为空: {path}.")
    for position in range(1, 4):
        sample = paths["model"] / "examples/input" / f"sample_{position}.jpg"
        if not sample.is_file():
            raise FileNotFoundError(f"缺少固定示例图片: {sample}.")
    directories = [f"{destination}/{name}" for name in ("models", "board", "examples/input")]
    run_command(["ssh", *ssh_options, board_host,
                 "mkdir -p " + " ".join(shlex.quote(path) for path in directories)])
    for folder, files in groups.items():
        remote = str(Path(destination) / folder) + "/"
        run_command(["scp", *ssh_options, *files, f"{board_host}:{remote}"])
    run_command(["ssh", *ssh_options, board_host,
                 "chmod 755 " + shlex.quote(f"{destination}/board/fastsam_board")])
    print(f"[下一步] 登录 {board_host},执行: bash {destination}/run.sh", flush=True)


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
    for name in ("tmp", "cache"):
        (work / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(work / "tmp"),
                      XDG_CACHE_HOME=str(work / "cache"), TORCH_HOME=str(work / "cache/torch"))
    paths = {"model": model, "repo": REPO_ROOT, "work": work,
             "dataset": Path(config["data"]["dataset_dir"]),
             "output": model / config["model"]["output_dir"]}
    print(f"[步骤] {Path(__file__).stem}; 工作目录: {work}", flush=True)
    upload(config, paths, config_path)


if __name__ == "__main__":
    main()
