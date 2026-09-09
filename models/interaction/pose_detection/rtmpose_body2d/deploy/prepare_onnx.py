"""生成 mtk_converter 8.16.0 可读取的 RTMPose 单文件 ONNX."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

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


def remove_safe_reshape_allowzero(model: onnx.ModelProto) -> int:
    """移除不改变当前模型语义但 MTK 导入器不支持的 allowzero 属性."""
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    removed = 0
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        allowzero = next(
            (item for item in node.attribute if item.name == "allowzero"),
            None)
        if allowzero is None:
            continue
        if allowzero.i == 1:
            shape_tensor = initializers.get(node.input[1])
            if shape_tensor is None:
                raise ValueError(
                    f"无法安全移除动态 Reshape allowzero=1: {node.name}")
            shape = numpy_helper.to_array(shape_tensor)
            if np.any(shape == 0):
                raise ValueError(
                    f"无法安全移除含 0 shape 的 Reshape allowzero=1: "
                    f"{node.name}, shape={shape.tolist()}")
        node.attribute.remove(allowzero)
        removed += 1
    return removed


def remove_rgb_to_bgr_prefix(model: onnx.ModelProto) -> int:
    """移除严格匹配的 RGB 到 BGR 前缀,兼容模型改为直接接收 BGR."""
    nodes = list(model.graph.node)
    if len(nodes) < 4:
        raise ValueError("RTMPose 图节点数量异常,无法识别通道重排前缀.")
    first, channel_gather, second, consumer = nodes[:4]
    if ([first.op_type, channel_gather.op_type, second.op_type] !=
            ["Reshape", "GatherND", "Reshape"] or
            first.input[0] != "image" or
            channel_gather.input[0] != first.output[0] or
            second.input[0] != channel_gather.output[0] or
            second.output[0] not in consumer.input):
        raise ValueError("RTMPose 前端不是预期的 Reshape/GatherND/Reshape.")
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    indices_tensor = initializers.get(channel_gather.input[1])
    if indices_tensor is None:
        raise ValueError("RTMPose 通道重排索引不是常量.")
    indices = numpy_helper.to_array(indices_tensor)
    if indices.shape != (3, 1) or indices.reshape(-1).tolist() != [2, 1, 0]:
        raise ValueError(
            f"RTMPose 通道重排索引异常: {indices.reshape(-1).tolist()}")
    replaced_input = second.output[0]
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == replaced_input:
                node.input[index] = "image"
    removed_names = {
        first.input[1], channel_gather.input[1], second.input[1],
        first.output[0], channel_gather.output[0], second.output[0],
    }
    for tensor in list(model.graph.initializer):
        if tensor.name in removed_names:
            model.graph.initializer.remove(tensor)
    for value in list(model.graph.value_info):
        if value.name in removed_names:
            model.graph.value_info.remove(value)
    del model.graph.node[:3]
    return 1


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
    reshape_count = remove_safe_reshape_allowzero(model)
    bgr_prefix_count = remove_rgb_to_bgr_prefix(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(output), save_as_external_data=False)
    reloaded = onnx.load(str(output), load_external_data=False)
    external_count = sum(
        tensor.data_location == onnx.TensorProto.EXTERNAL
        for tensor in reloaded.graph.initializer)
    if external_count:
        raise RuntimeError(f"兼容模型仍包含 {external_count} 个外部权重引用.")
    print(f"[OK] MTK 兼容 ONNX: IR={model.ir_version}, "
          f"opset={standard_opset}, 移除 Reshape allowzero "
          f"x{reshape_count}, 移除 RGB 到 BGR 前缀 "
          f"x{bgr_prefix_count}, {output}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    prepare_model(arguments.source, arguments.output)
