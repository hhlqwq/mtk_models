"""板端步骤读取 JSON 配置快照,不依赖主机或 PyYAML."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sys
import subprocess

from tools.evaluation import add_test_argument, is_test
from tools.process import run_command


def context(entry):
    """板端沿用原部署目录,快速结果仅追加运行编号的 _test 后缀."""
    root = Path(entry).resolve().parent.parent
    parser = argparse.ArgumentParser(description="执行当前板端独立步骤.")
    parser.add_argument("--config", type=Path, default=root / "config.json")
    parser.add_argument("--run-id", default=os.environ.get("EVAL_RUN_ID"))
    add_test_argument(parser)
    args = parser.parse_args()
    return paths(root, args.config, args.run_id)


def paths(root, config_path, run_id=None):
    """从上传的 JSON 快照计算本次结果位置."""
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_test() and not run_id.endswith("_test"):
        run_id += "_test"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("运行编号只允许字母、数字、下划线和连字符.")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    result_name = ("latest_test" if is_test() else "latest") if config["board"].get(
        "retain_latest", False) else run_id
    result = root / config["board"].get("results_dir", "results") / result_name
    return argparse.Namespace(root=root, config=config, run_id=run_id,
        dataset=Path(config["board"]["dataset_dir"]), result=result, work=result / "work")


def command(ctx, filename, *arguments, **kwargs):
    """执行板端程序或 Python 脚本,失败时保留原始错误."""
    path = ctx.root / "board" / filename
    prefix = [sys.executable] if path.suffix == ".py" else []
    run_command([*prefix, path, *arguments], **kwargs)


def configure_parser(parser, entry, bindings):
    """为公共指标和绘图工具提供板端配置默认值,保留原命令兼容性."""
    if not any(action.dest == "test" for action in parser._actions):
        add_test_argument(parser)
    if not any(action.dest == "config" for action in parser._actions):
        parser.add_argument("--config", type=Path)
    bootstrap = argparse.ArgumentParser(add_help=False)
    default_config = next(action.default for action in parser._actions if action.dest == "config")
    bootstrap.add_argument("--config", type=Path, default=default_config)
    bootstrap.add_argument("--run-id", default=os.environ.get("EVAL_RUN_ID"))
    options, _ = bootstrap.parse_known_args()
    if options.config is None or "-h" in sys.argv or "--help" in sys.argv:
        return
    ctx = paths(Path(entry).resolve().parent.parent, options.config, options.run_id)
    for action in parser._actions:
        if action.dest in bindings:
            action.default = bindings[action.dest](ctx)
            action.required = False


def write_int8_config(metadata, destination, input_shape, output_shape):
    """核对模型张量契约并生成 C++ 可直接读取的两行量化配置."""
    rows = []
    for name, shape in (("input", input_shape), ("output", output_shape)):
        info = metadata[name]
        scale, zero = float(info["scale"]), info["zero_point"]
        if (info["shape"] != shape or info["dtype"] != "int8" or
                not math.isfinite(scale) or scale <= 0 or type(zero) is not int or
                not -128 <= zero <= 127):
            raise ValueError(f"模型 {name} 张量契约无效: {info}.")
        rows.append(f"{scale:.17g},{zero}\n")
    destination.write_text("".join(rows), encoding="utf-8")


def infer_images(ctx, program, images, metadata, input_shape, output_shape):
    """仅调度单图 C++ 推理并保存记录,不在 Python 执行预处理或后处理."""
    if ctx.result.exists():
        raise FileExistsError(f"结果目录已存在: {ctx.result}.")
    if not images:
        raise ValueError("推理图片列表为空.")
    ctx.work.mkdir(parents=True)
    config = ctx.work / "runtime_config.csv"
    write_int8_config(metadata, config, input_shape, output_shape)
    (ctx.work / "logs").mkdir()
    print(f"[推理] {program}; {len(images)} 张图片; 结果目录: {ctx.result}", flush=True)
    with (ctx.work / "predictions.jsonl").open("w", encoding="utf-8") as records:
        for index, (name, image) in enumerate(images, 1):
            output = ctx.work / "raw" / f"{index:05d}"
            log_path = ctx.work / "logs" / f"{index:05d}.log"
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    command(ctx, program, "--model", ctx.root / "models/model_int8.dla",
                            "--config", config, "--image", image, "--output-dir", output,
                            "--quiet", quiet=True, stdout=log, stderr=subprocess.STDOUT)
            except (subprocess.CalledProcessError, OSError):
                print(log_path.read_text(encoding="utf-8"), file=sys.stderr, flush=True)
                raise
            timing = json.loads((output / "timing.json").read_text(encoding="utf-8"))
            if (timing.get("backend") != "cpp_neuron_runtime_hw" or
                    not math.isfinite(timing["npu_ms"]) or timing["npu_ms"] <= 0):
                raise ValueError(f"无效的 C++ 推理记录: {output}.")
            record = dict(image=name, directory=str(output.relative_to(ctx.work)), **timing)
            records.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.flush()
            if index % 50 == 0 or index == len(images):
                print(f"[推理] {index}/{len(images)}", flush=True)


def inference_records(work, expected):
    """核对本次 C++ 推理覆盖范围,避免旧输出、重复图片或越界路径混入评测."""
    records = {}
    for line in (work / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        name = row["image"]
        directory = Path(row["directory"])
        if (name in records or directory.is_absolute() or ".." in directory.parts or
                not directory.parts or directory.parts[0] != "raw" or
                row.get("backend") != "cpp_neuron_runtime_hw" or
                not math.isfinite(row["npu_ms"]) or row["npu_ms"] <= 0 or
                type(row["peak_rss_kib"]) is not int or row["peak_rss_kib"] <= 0):
            raise ValueError(f"无效或重复的推理记录: {name}.")
        records[name] = row
    if set(records) != set(expected):
        raise ValueError("推理记录与本次选定评测图片不一致.")
    return records
