"""板端步骤读取 JSON 配置快照,不依赖主机或 PyYAML."""

import argparse
from datetime import date, datetime
import json
import math
import os
from pathlib import Path
import re
import sys

from tools.evaluation import is_test


def context(entry, *, new_run=False):
    """读取板端配置,过程文件统一位于日期临时目录."""
    root = Path(entry).resolve().parent.parent
    parser = argparse.ArgumentParser(description="执行当前板端独立步骤.")
    parser.add_argument("--config", type=Path, default=root / "config.json")
    parser.add_argument("--run-id", default=os.environ.get("EVAL_RUN_ID"))
    args = parser.parse_args()
    return paths(args.config.resolve().parent, args.config, args.run_id, new_run=new_run)


def paths(root, config_path, run_id=None, *, new_run=False):
    """推理新建过程目录,后续单步默认复用当天同模型和范围的最近一次结果."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    stage = "compile" if is_test() else "evaluate"
    temporary = Path("/tmp/hailongcodex") / date.today().isoformat()
    prefix = f"{config['name']}_{stage}_"
    run_id = run_id or os.environ.get("EVAL_RUN_ID")
    if run_id is None and not new_run:
        latest = max((path for path in temporary.glob(prefix + "*") if path.is_dir()),
                     key=lambda path: path.stat().st_mtime, default=None)
        if latest is not None:
            run_id = latest.name[len(prefix):]
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("运行编号只允许字母、数字、下划线和连字符.")
    result = Path(os.environ.get("MTK_RUN_DIR", temporary / f"{config['name']}_{stage}_{run_id}"))
    if not result.resolve().is_relative_to(temporary.resolve()) or result.is_symlink():
        raise ValueError("过程目录必须位于当天 /tmp/hailongcodex 日期目录.")
    return argparse.Namespace(root=root, config=config, run_id=run_id,
        dataset=Path(config["board"]["dataset_dir"]), result=result, work=result / "work",
        summary=root / "results/summary.json")


def configure_parser(parser, entry, bindings, *, new_run=False):
    """为公共指标和绘图工具提供板端配置默认值,保留原命令兼容性."""
    default_config = Path(entry).resolve().parent.parent / "config.json"
    if not any(action.dest == "config" for action in parser._actions):
        parser.add_argument("--config", type=Path,
                            default=default_config if default_config.is_file() else None)
    bootstrap = argparse.ArgumentParser(add_help=False)
    default_config = next(action.default for action in parser._actions if action.dest == "config")
    bootstrap.add_argument("--config", type=Path, default=default_config)
    bootstrap.add_argument("--run-id", default=os.environ.get("EVAL_RUN_ID"))
    options, _ = bootstrap.parse_known_args()
    if options.config is None or "-h" in sys.argv or "--help" in sys.argv:
        return
    ctx = paths(options.config.resolve().parent, options.config, options.run_id, new_run=new_run)
    for action in parser._actions:
        if action.dest in bindings:
            action.default = bindings[action.dest](ctx)
            action.required = False
        elif action.dest == "run_id":
            action.default = ctx.run_id
            action.required = False


def infer_images(ctx, images, metadata, input_shape, output_shape, preprocess, postprocess):
    """用常驻 Python Runtime 处理图片,仅保存评分所需的输出与记录."""
    import cv2
    import numpy as np

    from tools.runtime import Runtime, peak_rss_kib, quantize

    if ctx.result.exists():
        raise FileExistsError(ctx.result)
    if not images:
        raise ValueError("推理图片列表为空.")
    for name, shape in (("input", input_shape), ("output", output_shape)):
        item = metadata[name]
        if (item["shape"] != shape or item["dtype"] != "int8"
                or not math.isfinite(item["scale"]) or item["scale"] <= 0
                or type(item["zero_point"]) is not int or not -128 <= item["zero_point"] <= 127):
            raise ValueError(f"模型 {name} 张量契约无效.")
    ctx.work.mkdir(parents=True)
    with Runtime(ctx.root / "models/model_int8.dla") as runtime:
        if tuple(map(len, runtime.sizes)) != (1, 1):
            raise ValueError("要求单输入单输出模型.")
        inputs, input_view = runtime.tensor(0, input_shape, np.int8,
                                            fill=metadata["input"]["zero_point"])
        outputs, output_view = runtime.tensor(0, output_shape, np.int8, output=True)
        with (ctx.work / "predictions.jsonl").open("w", encoding="utf-8") as records:
            for index, (name, path) in enumerate(images, 1):
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"无法读取图片: {path}.")
                np.copyto(input_view, quantize(preprocess(path), metadata["input"]["scale"],
                                               metadata["input"]["zero_point"]))
                milliseconds = runtime.infer([inputs], [outputs])
                values = (output_view.astype(np.float32) - metadata["output"]["zero_point"]) * metadata["output"]["scale"]
                output = ctx.work / "raw" / f"{index:05d}"
                output.mkdir(parents=True)
                postprocess(values, image, output)
                record = {"image": name, "directory": str(output.relative_to(ctx.work)),
                          "backend": "python_neuron_runtime_hw"}
                if not is_test():
                    record.update(npu_ms=milliseconds, peak_rss_kib=peak_rss_kib())
                records.write(json.dumps(record, ensure_ascii=False) + "\n")
                if index % 50 == 0 or index == len(images):
                    print(f"[推理] {ctx.config['name']} {index}/{len(images)}", flush=True)


def inference_records(work, expected):
    """核对本次 Python 推理覆盖范围,避免旧输出、重复图片或越界路径混入评测."""
    records = {}
    for line in (work / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        name = row["image"]
        directory = Path(row["directory"])
        if (name in records or directory.is_absolute() or ".." in directory.parts or
                not directory.parts or directory.parts[0] != "raw" or
                row.get("backend") != "python_neuron_runtime_hw"):
            raise ValueError(f"无效或重复的推理记录: {name}.")
        if not is_test() and (not math.isfinite(row["npu_ms"]) or row["npu_ms"] <= 0
                          or type(row["peak_rss_kib"]) is not int or row["peak_rss_kib"] <= 0):
            raise ValueError(f"评价性能记录无效: {name}.")
        records[name] = row
    if set(records) != set(expected):
        raise ValueError("推理记录与本次选定评测图片不一致.")
    return records
