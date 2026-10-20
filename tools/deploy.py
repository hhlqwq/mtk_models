"""主机独立步骤的配置、路径与命令工具,不分派模型业务."""

import argparse
import json
import os
import shlex
import shutil
from pathlib import Path
import sys

from tools.evaluation import add_test_argument, is_test
from tools.process import run_command
from tools.storage import REPO_ROOT, load_model_config, storage_roots


def context(entry, config_path=None, work_dir=None):
    """按入口定位模型配置,只在执行当前步骤时准备 build 目录."""
    model = Path(entry).resolve().parent.parent.parent
    config_path = Path(config_path or model / "config.yaml")
    config = load_model_config(config_path)
    output = Path(config["model"]["output_dir"])
    work = Path(work_dir) if work_dir else output / "build"
    for name in ("tmp", "cache"):
        (work / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(work / "tmp"),
                      XDG_CACHE_HOME=str(work / "cache"), TORCH_HOME=str(work / "cache/torch"))
    return argparse.Namespace(config=config, config_path=config_path, model=model,
        repo=REPO_ROOT, output=output, work=work,
        dataset=Path(config.get("data", {}).get("dataset_dir", output)),
        calibration=storage_roots()["dataset_root"] / config.get("data", {}).get("calibration_dir", "."))


def arguments(entry):
    """新独立步骤默认读取模型根目录 YAML,允许显式覆盖配置或工作目录."""
    parser = argparse.ArgumentParser(description="读取模型配置并执行当前独立步骤.")
    parser.add_argument("--config", type=Path,
                        default=Path(entry).resolve().parent.parent.parent / "config.yaml")
    parser.add_argument("--work-dir", type=Path)
    add_test_argument(parser)
    args = parser.parse_args()
    return context(entry, args.config, args.work_dir)


def configure_parser(parser, entry, bindings):
    """为已有参数入口绑定配置默认值,保留显式命令行参数的优先级."""
    parser.add_argument("--config", type=Path,
                        default=Path(entry).resolve().parent.parent.parent / "config.yaml")
    if not any(action.dest == "work_dir" for action in parser._actions):
        parser.add_argument("--work-dir", type=Path)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=Path(entry).resolve().parent.parent.parent / "config.yaml")
    bootstrap.add_argument("--work-dir", type=Path)
    options, _ = bootstrap.parse_known_args()
    if "-h" in sys.argv or "--help" in sys.argv:
        return None
    ctx = context(entry, options.config, options.work_dir)
    for action in parser._actions:
        if action.dest == "work_dir":
            action.default = ctx.work
        if action.dest in bindings:
            action.default = bindings[action.dest](ctx)
            action.required = False
    return ctx


def reference_path(ctx):
    """快速参考单独命名,不覆盖全量报告."""
    return ctx.work / ("onnx_reference_test.json" if is_test() else "onnx_reference.json")


def compile_dla(ctx, source, target, extra=()):
    """仅在编译器子进程设置 SDK 动态库路径,避免污染 Python 环境."""
    ncc = Path(os.environ["NCC_ROOT"])
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(ncc / "lib")
    run_command([ncc / "bin/ncc-tflite", "--arch=mdla5.3", "--suppress-output",
                 "--disallow-bridge", *extra, source, "-o", target], env=environment)
    if not Path(target).is_file() or not Path(target).stat().st_size:
        raise FileNotFoundError(f"DLA 未生成: {target}.")
    print(f"[DLA] 最终模型: {Path(target).resolve()}", flush=True)


def upload(ctx, model_files, programs, extra_files=()):
    """上传当前模型部署包,快速与全量共用原部署目录,参考报告分别保存."""
    snapshot = ctx.work / "config.json"
    snapshot.write_text(json.dumps(ctx.config, ensure_ascii=False, indent=2))
    samples = ctx.work / "samples.json"
    if ctx.config["name"] not in ("yamnet", "deepseek_r1_distill_qwen_1_5b", "minicpm5_2b"):
        run_command([sys.executable, ctx.repo / "tools/prepare_examples.py",
                     "--model", ctx.config["name"], "--dataset-root", ctx.dataset,
                     "--output", samples])
    board_source = ctx.model / "deploy/board"
    code = [path for path in board_source.iterdir()
            if path.is_file() and path.suffix in (".py", ".json")]
    for name in ("render_examples.py", "summarize_board_result.py", "model_size.py"):
        if not (board_source / name).exists():
            code.append(ctx.repo / "tools" / name)
    reference = reference_path(ctx)
    if reference.is_file():
        model_files = [*model_files, reference]
    groups = {
        "": [ctx.model / "deploy/run.sh", snapshot],
        "models": model_files,
        "board": [*code, *programs],
        "board/tools": [ctx.repo / "tools" / name for name in
            ("__init__.py", "board.py", "evaluation.py", "files.py", "process.py", "coco.py")],
        "examples/input": [*sorted((ctx.model / "examples/input").glob("sample_*"))],
    }
    if samples.is_file():
        groups["examples/input"].append(samples)
    for folder, path in extra_files:
        groups.setdefault(folder, []).append(path)
    for files in groups.values():
        for path in files:
            if not Path(path).is_file() or not Path(path).stat().st_size:
                raise FileNotFoundError(f"上传文件缺失或为空: {path}.")
    host = os.environ["MTK_BOARD_HOST"]
    destination = ctx.config["board"]["deploy_dir"]
    options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    directories = [str(Path(destination) / name) for name in groups]
    run_command(["ssh", *options, host, "mkdir -p " +
                 " ".join(shlex.quote(path) for path in directories)])
    for folder, files in groups.items():
        if files:
            run_command(["scp", *options, *files, f"{host}:{destination}/{folder}/"])
    if programs:
        run_command(["ssh", *options, host, "chmod 755 " + " ".join(
            shlex.quote(str(Path(destination) / "board" / Path(path).name)) for path in programs)])
    flag = " -test" if is_test() else ""
    print(f"[下一步] 在开发板执行: bash {destination}/run.sh{flag}", flush=True)
