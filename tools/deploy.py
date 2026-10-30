"""主机独立步骤的配置、路径与命令工具,不分派模型业务."""

import argparse
import json
import os
import shlex
import shutil
from pathlib import Path
import sys

from tools.evaluation import is_test
from tools.process import run_command
from tools.storage import REPO_ROOT, load_model_config, storage_roots


def context(entry, config_path=None, work_dir=None):
    """按入口定位模型配置,只在执行当前步骤时准备 build 目录."""
    model = Path(entry).resolve().parent.parent.parent
    config_path = Path(config_path or model / "config.yaml")
    config = load_model_config(config_path)
    output = Path(config["model"]["output_dir"])
    work = Path(work_dir) if work_dir else output / "build"
    if (os.environ.get("MTK_EVALUATION_WORK") == "1"
            or Path(entry).stem.startswith("evaluate")):
        from datetime import date

        stage = "compile" if (os.environ.get("MTK_PREPARE_OUTPUT_REFERENCE") == "1"
                              or os.environ.get("MTK_COMPILE_ACCURACY") == "1") else "evaluate"
        work = Path("/tmp/hailongcodex") / date.today().isoformat() / config["name"] / stage
    work.mkdir(parents=True, exist_ok=True)
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


def upload(ctx, model_files, extra_files=(), *, multi_file=False):
    """上传当前模型部署包并提示板端入口,不执行推理或准备正式评价参考."""
    ctx.config["deployment"] = {"model_directory": "models" if multi_file else "."}
    snapshot = ctx.work / "config.json"
    snapshot.write_text(json.dumps(ctx.config, ensure_ascii=False, indent=2))
    samples = ctx.work / "samples.json"
    if ctx.config["name"] == "yamnet":
        samples = ctx.model / "examples/input/samples.json"
    if ctx.config["name"] not in ("yamnet", "deepseek_r1_distill_qwen_1_5b", "minicpm5_2b"):
        run_command([sys.executable, ctx.repo / "tools/prepare_examples.py",
                     "--model", ctx.config["name"], "--dataset-root", ctx.dataset,
                     "--output", samples])
    board_source = ctx.model / "deploy/board"
    code = [path for path in board_source.iterdir()
            if path.is_file() and path.suffix in (".py", ".json")]
    for name in ("render_examples.py",):
        if not (board_source / name).exists():
            code.append(ctx.repo / "tools" / name)
    groups = {
        "": [ctx.model / "deploy/compile.sh", ctx.model / "deploy/evaluate.sh", snapshot],
        "build/reference": [] if multi_file else [path for path in model_files if path.suffix != ".dla"],
        "models": model_files if multi_file else [],
        "board": [*code, ctx.repo / "tools/summary.py", ctx.repo / "tools/compile_accuracy.py"],
        "board/tools": [ctx.repo / "tools" / name for name in
            ("__init__.py", "board.py", "evaluation.py", "files.py", "coco.py", "runtime.py", "output_metrics.py")],
        "examples/input": [*sorted((ctx.model / "examples/input").glob("sample_*"))],
    }
    if not multi_file:
        groups[""].extend(path for path in model_files if path.suffix == ".dla")
    if ctx.config["name"] in ("yolov5s", "yolov8n", "fastsam"):
        groups["board/tools"].append(ctx.repo / "tools/boxes.py")
    if ctx.config["name"] in ("yolov5s", "yolov8n"):
        groups["board/tools"].append(ctx.repo / "tools/detection.py")
    if samples.is_file():
        groups["examples/input"].append(samples)
    for folder, path in extra_files:
        groups.setdefault(folder, []).append(path)
    for files in groups.values():
        for path in files:
            if not Path(path).is_file() or (not Path(path).stat().st_size and Path(path).name != "__init__.py"):
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
    if multi_file:
        prune_models(ctx, [path.name for path in model_files])
    print("[待板端检查] 上传完成,尚未执行板端推理或生成展示示例.", flush=True)
    print(f"[板端入口] 请登录开发板后执行: cd -- {shlex.quote(destination)} && bash compile.sh",
          flush=True)
    print("[评价入口] 准备同协议参考和必要输入后,在板端部署目录执行 bash evaluate.sh.",
          flush=True)


def deployment_artifacts(ctx):
    """按当前部署契约选择实际使用的 DLA,不把历史残留文件计入参考."""
    name = ctx.config["name"]
    root = ctx.output
    if name in ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b"):
        from tools.llm.artifacts import dla_relative_path

        root = ctx.output / "models"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        names = [dla_relative_path(manifest, phase) for phase in ("prompt", "decode")]
    elif name == "mit_bevfusion":
        root = ctx.output / "models"
        contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
        names = [item["file"] for item in contract["stages"].values() if item["backend"] == "mtk_npu"]
    elif name == "whisper_tiny":
        contract = json.loads((root / "build/runtime_contract.json").read_text(encoding="utf-8"))
        names = [contract[stage]["file"] for stage in ("encoder", "decoder_step")]
    elif name == "yamnet":
        names = [f"model_{ctx.config['conversion']['precision']}.dla"]
    else:
        names = ["model_int8.dla"]
    if not names or len(set(names)) != len(names):
        raise ValueError("部署契约没有唯一的有效 DLA 列表.")
    files = {}
    for name in names:
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"部署 DLA 缺失、越界或为链接: {name}.")
        files[name] = path
    return files


def transfer_files(ctx, files):
    """按明确的相对目标上传必要文件,不传过程目录或隐含递归资源."""
    from pathlib import PurePosixPath

    if not files:
        raise ValueError("上传文件清单不能为空.")
    for target, source in files.items():
        relative = PurePosixPath(target)
        if relative.is_absolute() or ".." in relative.parts or "\\" in target:
            raise ValueError("上传目标必须位于当前模型目录内.")
        if not Path(source).is_file() or Path(source).is_symlink():
            raise FileNotFoundError(f"上传文件缺失或为链接: {source}.")
    host = os.environ["MTK_BOARD_HOST"]
    root = ctx.config["board"]["deploy_dir"].rstrip("/")
    folders = sorted({str(Path(root) / Path(name).parent) for name in files}
                     | {root + "/build/reference"})
    options = ["-o", "BatchMode=yes"]
    run_command(["ssh", *options, host, "mkdir -p -- " + " ".join(map(shlex.quote, folders))])
    for target, source in files.items():
        run_command(["scp", *options, source, f"{host}:{root}/{target}"])


def prune_models(ctx, allowed):
    """上传完整后仅清理当前模型 models/ 内未列入运行包的旧产物."""
    root = ctx.config["board"]["deploy_dir"]
    script = "\n".join((
        "from pathlib import Path",
        f"root = Path({root!r})",
        f"allowed = {sorted(allowed)!r}",
        "if not root.is_absolute() or root == Path('/') or root.is_symlink(): raise ValueError('部署根目录不安全')",
        "folder = root / 'models'",
        "if folder.is_symlink() or folder.resolve().parent != root.resolve(): raise ValueError('模型目录越界')",
        "expected = {folder / name for name in allowed}",
        "if not expected or any(not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()) for path in expected): raise ValueError('新部署包不完整或越界,不清理旧文件')",
        "existing = list(folder.rglob('*'))",
        "if any(path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()) for path in existing): raise ValueError('旧目录含链接或越界文件,不清理')",
        "old = [path for path in existing if path.is_file() and path not in expected]",
        "for path in old: path.unlink()",
        "for path in sorted((path for path in existing if path.is_dir()), key=lambda path: len(path.parts), reverse=True):",
        "    if not any(path.iterdir()): path.rmdir()",
        "print(f'[模型目录] 清理 {len(old)} 个旧文件,保留当前运行包')",
    ))
    run_command(["ssh", "-o", "BatchMode=yes", os.environ["MTK_BOARD_HOST"],
                 "python3 -c " + shlex.quote(script)])


def upload_llm(ctx, package):
    """将多文件模型资源与板端代码分开放置,不删除已有结果或参考."""
    package = Path(package)
    config = json.loads((package / "config.json").read_text(encoding="utf-8"))
    config["deployment"] = {"model_directory": "models"}
    snapshot = ctx.work / "config.json"
    snapshot.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    files = {"config.json": snapshot}
    for source in sorted(package.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(package)
        if relative == Path("config.json"):
            continue
        folder = relative.parts[0]
        target = relative if folder in ("board", "examples", "compile.sh", "evaluate.sh") else Path("models") / relative
        files[target.as_posix()] = source
    # 参考单独上传,不会因模型代码刷新而被删除.
    transfer_files(ctx, files)
    prune_models(ctx, [name.removeprefix("models/") for name in files if name.startswith("models/")])


def upload_evaluation(deploy):
    """上传本次全量浮点结果和必要输入,不量化、编译或运行板端模型."""
    from tools.evaluation import read_accuracy
    from tools.files import read_json, sha256

    ctx = context(Path(deploy) / "host/evaluate_onnx.py")
    name = ctx.config["name"]
    files = {}
    if name in ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b"):
        mode = "mmlu1000"
        data = ctx.work / "evaluation" / mode
        names = ("mmlu/protocol.json", "mmlu/cases.json", "mmlu/mmlu_inputs.npz",
                 "mmlu/board_input_contract.json", "quality/board_quality_protocol.json",
                 "quality/board_inputs.npz", "quality/board_input_contract.json",
                 "reference/mmlu/mmlu_accuracy.json", "quality/pytorch_quality.json")
        read_accuracy(read_json(data / "reference/mmlu/mmlu_accuracy.json"), name, "pytorch")
        files.update({"build/reference/full/" + filename: data / filename for filename in names})
    elif name == "mit_bevfusion":
        source = ctx.work / "onnx_reference.json"
        read_accuracy(read_json(source), name, "onnx")
        files.update({"build/reference/full.json": source,
                      "build/reference/full_samples.json": ctx.work / "samples.json",
                      "build/reference/full_ground_truth.json": ctx.work / "ground_truth.json"})
    else:
        source = (ctx.work / "evaluation/full/onnx/summary.json" if name == "yamnet"
                  else reference_path(ctx))
        read_accuracy(read_json(source), name, "onnx")
        files["build/reference/full.json"] = source
        if name == "yamnet":
            files["build/reference/full_detail.json"] = ctx.work / "fp32_reference.json"
    main_source = (data / "reference/mmlu/mmlu_accuracy.json"
                   if name in ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b") else source)
    reference = read_json(main_source)
    artifacts = deployment_artifacts(ctx)
    reference["dla_artifacts"] = {name: sha256(path) for name, path in artifacts.items()}
    target = ctx.work / "full_reference.json"
    target.write_text(json.dumps(reference, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    files["build/reference/full.json"] = target
    transfer_files(ctx, files)
    print(f"[全量参考上传完成] {name}; 待板端 evaluate.sh.", flush=True)


def main():
    """共享上传步骤由 Shell 入口直接调用,不分派模型推理或评分."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload-reference", type=Path, required=True)
    args = parser.parse_args()
    if is_test():
        raise ValueError("全量参考上传只能从 evaluate 入口调用.")
    upload_evaluation(args.upload_reference)


if __name__ == "__main__":
    main()
