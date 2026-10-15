"""编译 FastSAM DLA,在 Docker 内执行."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess

import yaml

from tools.storage import REPO_ROOT, load_model_config


def run_command(command: list, **kwargs) -> None:
    """显示本步骤实际调用的命令,失败时立即停止."""
    arguments = [str(item) for item in command]
    print(f"[执行] {shlex.join(arguments)}", flush=True)
    subprocess.run(arguments, check=True, **kwargs)


def compile_model(config: dict, paths: dict, config_path: Path) -> None:
    """调用 Neuron 编译器生成 Genio 720 使用的 DLA."""
    toolchain_path = paths["repo"] / "configs/toolchains/neuropilot_8_0_11.yaml"
    toolchain = yaml.safe_load(toolchain_path.read_text(encoding="utf-8"))
    default_ncc = Path(toolchain["container_sdk_root"]) / "neuron_sdk/host"
    ncc = Path(os.environ.get("NCC_ROOT", str(default_ncc)))
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = f"{ncc / 'lib'}:{environment.get('LD_LIBRARY_PATH', '')}"
    run_command([ncc / "bin/ncc-tflite", "--arch=mdla5.3", "--suppress-output",
                 "--disallow-bridge", paths["work"] / "model_int8.tflite",
                 "-o", paths["output"] / "model_int8.dla"], env=environment)
    print(f"[DLA] 文件路径: {(paths['output'] / 'model_int8.dla').resolve()}", flush=True)


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
    compile_model(config, paths, config_path)


if __name__ == "__main__":
    main()
