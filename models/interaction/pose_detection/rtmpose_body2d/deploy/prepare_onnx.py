"""生成 mtk_converter 8.16.0 可读取的 RTMPose 单文件 ONNX."""

import argparse
from pathlib import Path

import onnx

MAX_SUPPORTED_IR = 8
MAX_SUPPORTED_OPSET = 18


def prepare_model(source: Path, output: Path) -> None:
    """合并外部权重并在语义不变时降低 ONNX IR 版本."""
    model = onnx.load(str(source), load_external_data=True)
    opsets = {(item.domain or "ai.onnx"): item.version
              for item in model.opset_import}
    standard_opset = opsets.get("ai.onnx", 0)
    print(f"[INFO] Qualcomm RTMPose: IR={model.ir_version}, opset={opsets}")
    if standard_opset > MAX_SUPPORTED_OPSET:
        raise ValueError(
            f"ONNX opset {standard_opset} 超出 MTK 支持上限 "
            f"{MAX_SUPPORTED_OPSET}, 不能只改版本号.")
    if model.ir_version > MAX_SUPPORTED_IR:
        model.ir_version = MAX_SUPPORTED_IR
    onnx.checker.check_model(model)
    onnx.save(model, str(output), save_as_external_data=False)
    reloaded = onnx.load(str(output), load_external_data=False)
    external_count = sum(
        tensor.data_location == onnx.TensorProto.EXTERNAL
        for tensor in reloaded.graph.initializer)
    if external_count:
        raise RuntimeError(f"兼容模型仍包含 {external_count} 个外部权重引用.")
    print(f"[OK] MTK 兼容 ONNX: IR={model.ir_version}, "
          f"opset={standard_opset}, {output}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    prepare_model(arguments.source, arguments.output)
