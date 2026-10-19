"""将自行导出的 Qwen2 ONNX 分片转换为 FP32 TFLite,编译由独立步骤完成."""

import argparse
import json
from pathlib import Path

import mtk_converter

from tools.deploy import configure_parser


def convert_model(args):
    """逐个转换静态分片,中间模型只保存在 build 目录."""
    manifest = json.loads((args.models / "export_manifest.json").read_text())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for index, graph in enumerate(manifest["graphs"], 1):
        name = graph["name"]
        print(f"[转换 {index}/{len(manifest['graphs'])}] {name}", flush=True)
        converter = mtk_converter.OnnxConverter.from_model_proto_file(
            str(args.models / f"{name}.onnx"))
        converter.quantize = False
        target = args.work_dir / f"{name}.tflite"
        converter.convert_to_tflite(str(target))
        print(f"[TFLite] {target}", flush=True)


def parse_args():
    """直接读取模型配置,也允许显式指定已有静态图目录."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    configure_parser(parser, __file__, {"models": lambda ctx: ctx.output})
    return parser.parse_args()


if __name__ == "__main__":
    convert_model(parse_args())
