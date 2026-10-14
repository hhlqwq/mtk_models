"""上传 FastSAM 模型、程序与配置,在 89 主机执行."""

import json
import re
import argparse
from datetime import date
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import yaml


def run_command(command: list, **kwargs) -> None:
    """显示本步骤实际调用的命令,失败时立即停止."""
    arguments = [str(item) for item in command]
    print(f"[执行] {shlex.join(arguments)}", flush=True)
    subprocess.run(arguments, check=True, **kwargs)


def upload(config: dict, paths: dict, config_path: Path) -> None:
    """准备少量示例并上传部署文件,不在板端启动推理."""
    container = config["build"]["container"]
    board = config["board"]
    destination = board["deploy_dir"]
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", destination) or any(
            part in ("", ".", "..") for part in destination.split("/")[1:]):
        raise ValueError("board.deploy_dir 必须是无空格、空段或上级引用的绝对目录.")
    ssh_options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    examples = paths["work"] / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    run_command(["docker", "exec", "-e", "MTK_MODEL_STORAGE_ROOT", "-e", "MTK_DATASET_ROOT", "-e", "PYTHONDONTWRITEBYTECODE=1", container,
                 "python", paths["repo"] / "tools/prepare_examples.py", "--model",
                 "fastsam", "--dataset-root", paths["dataset"],
                 "--output", examples / "samples.json"])
    run_command(["docker", "cp", f"{container}:{examples / 'samples.json'}", examples / "samples.json"])
    reference = paths["work"] / "onnx_reference.json"
    run_command(["docker", "cp",
                 f"{container}:{paths['work'] / 'onnx_accuracy/summary.json'}", reference])
    report = json.loads(reference.read_text(encoding="utf-8"))
    if (report.get("status") != "complete" or report.get("model") != "fastsam"
            or report.get("metric") != "segm AR@100" or report.get("samples") != 5000):
        raise ValueError("缺少完整的 FastSAM 同协议 ONNX 参考精度,请先执行 evaluate_onnx.py.")
    # 板端读取标准 JSON,避免为了配置新增 PyYAML 依赖.
    snapshot = paths["work"] / "config.json"
    snapshot.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    deploy = paths["model"] / "deploy"
    groups = {
        "": [deploy / "run.sh", snapshot],
        "models": [paths["output"] / "model_int8.dla",
                   paths["output"] / "runtime_config.csv", reference],
        "board": [deploy / "board/infer.py", deploy / "board/full_accuracy_board.py",
                  deploy / "board/render.py", deploy / "board/summarize.py",
                  deploy / "board/accuracy_protocol.json", paths["work"] / "fastsam_board",
                  *[paths["repo"] / "tools" / name for name in
                    ("render_examples.py", "summarize_board_result.py", "model_size.py")]],
        "examples/input": [examples / "samples.json",
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
    run_command(["ssh", *ssh_options, board["host"],
                 "mkdir -p " + " ".join(shlex.quote(path) for path in directories)])
    for folder, files in groups.items():
        remote = str(Path(destination) / folder) + "/"
        run_command(["scp", *ssh_options, *files, f"{board['host']}:{remote}"])
    run_command(["ssh", *ssh_options, board["host"],
                 "chmod 755 " + shlex.quote(f"{destination}/board/fastsam_board")])
    print(f"[下一步] 登录 {board['host']},执行: bash {destination}/run.sh", flush=True)


def main() -> None:
    """直接读取 YAML 配置并执行本文件对应的操作."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "config.yaml")
    parser.add_argument("--work-dir", type=Path, help="跨天执行时复用原来的中间目录.")
    args = parser.parse_args()
    config_path = args.config.resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[6] / "tools"))
    from storage import load_model_config
    config = load_model_config(config_path)
    model = Path(__file__).resolve().parents[2]
    work = args.work_dir or Path(config["build"]["work_root"]) / os.environ.get("FASTSAM_RUN_DATE", date.today().isoformat()) / "fastsam"
    for name in ("tmp", "cache"):
        (work / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(work / "tmp"),
                      XDG_CACHE_HOME=str(work / "cache"), TORCH_HOME=str(work / "cache/torch"))
    paths = {"model": model, "repo": model.parents[3], "work": work,
             "dataset": Path(config["data"]["dataset_dir"]),
             "output": model / config["model"]["output_dir"]}
    print(f"[步骤] {Path(__file__).stem}; 工作目录: {work}", flush=True)
    upload(config, paths, config_path)


if __name__ == "__main__":
    main()
