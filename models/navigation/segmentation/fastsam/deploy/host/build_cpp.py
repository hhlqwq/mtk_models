"""交叉编译 FastSAM 板端 C++ 程序,在 89 主机执行."""

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


def build_cpp(config: dict, paths: dict, config_path: Path) -> None:
    """在 89 主机交叉编译 AArch64 板端推理程序."""
    build = config["build"]
    toolchain = Path(build["cpp_toolchain_root"])
    source = toolchain / "opencv-4.9.0/opencv-4.9.0"
    libraries = toolchain / "genio720-libs"
    includes = [toolchain / "opencv-4.9.0/build-aarch64-headers",
                source / "modules/core/include", source / "modules/imgproc/include",
                source / "modules/imgcodecs/include", Path(build["neuron_include"])]
    run_command([build["cross_cxx"], "-std=c++20", "-O3", "-DNDEBUG", "-Wall",
                 "-Wextra", "-Wpedantic", *[f"-I{path}" for path in includes],
                 paths["model"] / "deploy/board/fastsam_board.cpp",
                 libraries / "libneuronusdk_runtime.mtk.so.8",
                 libraries / "libopencv_imgcodecs.so.409",
                 libraries / "libopencv_imgproc.so.409",
                 libraries / "libopencv_core.so.409", "-Wl,--allow-shlib-undefined",
                 "-pthread", "-ldl", "-o", paths["work"] / "fastsam_board"])


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
    build_cpp(config, paths, config_path)


if __name__ == "__main__":
    main()
