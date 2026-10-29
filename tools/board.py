"""板端步骤读取 JSON 配置快照,不依赖主机或 PyYAML."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

from tools.evaluation import is_test


def context(entry, *, new_run=False):
    """读取板端配置,过程文件固定保存在当前模型的 build/work 目录."""
    root = Path(entry).resolve().parent.parent
    parser = argparse.ArgumentParser(description="执行当前板端独立步骤.")
    parser.add_argument("--config", type=Path, default=root / "config.json")
    parser.add_argument("--run-id", default=os.environ.get("EVAL_RUN_ID"))
    args = parser.parse_args()
    return paths(args.config.resolve().parent, args.config, args.run_id, new_run=new_run)


def paths(root, config_path, run_id=None, *, new_run=False):
    """每次推理覆盖当前模型的 build/work,后续评分和展示直接复用该目录."""
    root = Path(root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    run_id = run_id or os.environ.get("EVAL_RUN_ID")
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("运行编号只允许字母、数字、下划线和连字符.")
    result = root / "build/work"
    if result.is_symlink() or result.resolve() != result:
        raise ValueError("模型 work 目录不能是符号链接或指向部署目录之外.")
    models = root / config["deployment"]["model_directory"]
    if not models.resolve().is_relative_to(root) or models.is_symlink():
        raise ValueError("部署模型目录越界或为符号链接.")
    if new_run:
        from tools.evaluation import read_accuracy
        from tools.files import read_json, sha256

        filename = "subset.json" if is_test() else "full.json"
        reference = read_json(root / "build/reference" / filename)
        source = reference["evaluation"]["backend"]
        if source not in ("onnx", "pytorch"):
            raise ValueError("板端必须使用主机生成的浮点参考.")
        read_accuracy(reference, config["name"], source)
        if not reference["dla_artifacts"]:
            raise ValueError("浮点参考未绑定 DLA.")
        for name, digest in reference["dla_artifacts"].items():
            artifact = models / name
            if not artifact.resolve().is_relative_to(models.resolve()) or sha256(artifact) != digest:
                raise ValueError("板端 DLA 与当前浮点参考不匹配.")
    if new_run and result.exists():
        if not result.is_dir():
            raise ValueError("模型 work 路径必须是目录.")
        print(f"[过程目录] 覆盖本模型上次中间结果: {result}", flush=True)
        shutil.rmtree(result)
    return argparse.Namespace(root=root, config=config, run_id=run_id,
        dataset=Path(config["board"]["dataset_dir"]), result=result, work=result,
        models=root / config["deployment"]["model_directory"],
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
    from tools.output_metrics import OutputComparison

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
    comparison = OutputComparison(ctx)
    with Runtime(ctx.root / "model_int8.dla") as runtime:
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
                floating_input = preprocess(path)
                np.copyto(input_view, quantize(floating_input, metadata["input"]["scale"],
                                               metadata["input"]["zero_point"]))
                milliseconds = runtime.infer([inputs], [outputs])
                values = (output_view.astype(np.float32) - metadata["output"]["zero_point"]) * metadata["output"]["scale"]
                output = ctx.work / "raw" / f"{index:05d}"
                output.mkdir(parents=True)
                postprocess(values, image, output)
                comparison.update([floating_input], {"output": values},
                                  sample_id=name, sample_files=[path])
                record = {"image": name, "directory": str(output.relative_to(ctx.work)),
                          "backend": "python_neuron_runtime_hw"}
                if not is_test():
                    record.update(npu_ms=milliseconds, peak_rss_kib=peak_rss_kib())
                records.write(json.dumps(record, ensure_ascii=False) + "\n")
                if index % 50 == 0 or index == len(images):
                    print(f"[推理] {ctx.config['name']} {index}/{len(images)}", flush=True)
    comparison.finish()


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
