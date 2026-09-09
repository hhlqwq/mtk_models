"""生成 mtk_converter 8.16.0 可读取的 RTMPose 单文件 ONNX."""

import argparse
from pathlib import Path

import onnx

MAX_SUPPORTED_IR = 8
MAX_SUPPORTED_OPSET = 18


def validate_target_schemas(model: onnx.ModelProto) -> None:
    """确认标准 ONNX 节点在 opset 18 中存在且属性受支持."""
    for node in model.graph.node:
        domain = node.domain or ""
        if domain not in {"", "ai.onnx"}:
            continue
        schema = onnx.defs.get_schema(
            node.op_type, MAX_SUPPORTED_OPSET, domain)
        supported_attributes = set(schema.attributes)
        unknown_attributes = sorted(
            attribute.name for attribute in node.attribute
            if attribute.name not in supported_attributes)
        if unknown_attributes:
            raise ValueError(
                f"{node.name or node.op_type} 在 opset "
                f"{MAX_SUPPORTED_OPSET} 不支持属性: {unknown_attributes}")


def prepare_model(source: Path, output: Path) -> None:
    """合并外部权重并在语义不变时降低 ONNX IR 版本."""
    model = onnx.load(str(source), load_external_data=True)
    opsets = {(item.domain or "ai.onnx"): item.version
              for item in model.opset_import}
    standard_opset = opsets.get("ai.onnx", 0)
    print(f"[INFO] Qualcomm RTMPose: IR={model.ir_version}, opset={opsets}")
    if standard_opset > MAX_SUPPORTED_OPSET:
        validate_target_schemas(model)
        for item in model.opset_import:
            if (item.domain or "ai.onnx") == "ai.onnx":
                item.version = MAX_SUPPORTED_OPSET
        standard_opset = MAX_SUPPORTED_OPSET
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
