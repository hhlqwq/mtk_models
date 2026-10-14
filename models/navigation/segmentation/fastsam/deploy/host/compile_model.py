"""编译 FastSAM DLA,在 Docker 内执行."""

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


def compile_model(config: dict, paths: dict, config_path: Path) -> None:
    """调用 Neuron 编译器生成 Genio 720 使用的 DLA."""
    ncc = Path(config["build"]["ncc_root"])
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = f"{ncc / 'lib'}:{environment.get('LD_LIBRARY_PATH', '')}"
    run_command([ncc / "bin/ncc-tflite", "--arch=mdla5.3", "--suppress-output",
                 "--disallow-bridge", paths["output"] / "model_int8.tflite",
                 "-o", paths["output"] / "model_int8.dla"], env=environment)


def main() -> None:
    """直接读取 YAML 配置并执行本文件对应的操作."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "config.yaml")
    parser.add_argument("--work-dir", type=Path, help="跨天执行时复用原来的中间目录.")
    parser.add_argument("--inside-container", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = args.config.resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[6] / "tools"))
    from storage import load_model_config
    config = load_model_config(config_path)
    model = Path(__file__).resolve().parents[2]
    work = args.work_dir or Path(config["build"]["work_root"]) / os.environ.get("FASTSAM_RUN_DATE", date.today().isoformat()) / "fastsam"
    work = work.resolve()
    if not args.inside_container and not Path("/.dockerenv").exists():
        container = config["build"]["container"]
        run_command(["docker", "exec", "-e", "MTK_MODEL_STORAGE_ROOT", "-e", "MTK_DATASET_ROOT", container, "python", Path(__file__).resolve(),
                     "--inside-container", "--config", config_path, "--work-dir", work])
        return
    for name in ("tmp", "cache"):
        (work / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(work / "tmp"),
                      XDG_CACHE_HOME=str(work / "cache"), TORCH_HOME=str(work / "cache/torch"))
    paths = {"model": model, "repo": model.parents[3], "work": work,
             "dataset": Path(config["data"]["dataset_dir"]),
             "output": model / config["model"]["output_dir"]}
    print(f"[步骤] {Path(__file__).stem}; 工作目录: {work}", flush=True)
    compile_model(config, paths, config_path)


if __name__ == "__main__":
    main()
