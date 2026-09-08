"""将 Qualcomm ViT FP32 ONNX 降级为 mtk_converter 8.16.0 可接受的版本.

Qualcomm v0.61.0 导出为 IR v10 / opset 21, 且使用 opset 20 新增的 Gelu
算子; mtk_converter (onnx 1.13.1) 要求 IR v3..v8 且 opset <= 18.
本脚本做三项等价改写:

1. Gelu 展开为 TFLite 可导出的标准 tanh 近似子图;
2. ai.onnx opset 21 -> 17 (LayerNormalization 所需的最低 opset,
   其余算子 schema 在 17..21 之间无变化);
3. 删除常量 shape 且不含 0 的 Reshape allowzero=1 属性；
4. ir_version -> 8.

模型无需降级 (IR<=8 且 opset<=18 且无 Gelu) 时保持原样退出.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

MAX_SUPPORTED_OPSET = 18
TARGET_OPSET = 17
TARGET_IR = 8


def scalar_initializer(name: str, value: float):
    """构造 float32 标量 initializer."""
    return numpy_helper.from_array(np.array(value, dtype=np.float32), name)


def expand_gelu(node, index: int) -> tuple[list, list]:
    """把单个 Gelu 节点展开为标准 tanh 近似子图."""
    x = node.input[0]
    y = node.output[0]
    prefix = f"gelu_{index}"
    half = f"{prefix}_half"
    one = f"{prefix}_one"
    t2 = f"{prefix}_t2"
    t3 = f"{prefix}_t3"
    k0 = f"{prefix}_k0"
    k1 = f"{prefix}_k1"
    three = f"{prefix}_three"
    x3 = f"{prefix}_x3"
    t1 = f"{prefix}_t1"
    inner = f"{prefix}_inner"
    scaled = f"{prefix}_scaled"
    tanh_out = f"{prefix}_tanh"
    inits = [
        scalar_initializer(half, 0.5),
        scalar_initializer(one, 1.0),
        scalar_initializer(k0, math.sqrt(2.0 / math.pi)),
        scalar_initializer(k1, 0.044715),
        scalar_initializer(three, 3.0),
    ]
    nodes = [
        helper.make_node("Pow", [x, three], [x3], name=f"{prefix}_pow3"),
        helper.make_node("Mul", [x3, k1], [t1], name=f"{prefix}_mul_k1"),
        helper.make_node("Add", [x, t1], [inner], name=f"{prefix}_add_x"),
        helper.make_node("Mul", [inner, k0], [scaled], name=f"{prefix}_mul_k0"),
        helper.make_node("Tanh", [scaled], [tanh_out], name=f"{prefix}_tanh"),
        helper.make_node("Add", [tanh_out, one], [t2], name=f"{prefix}_one_plus"),
        helper.make_node("Mul", [x, t2], [t3], name=f"{prefix}_x_mul"),
        helper.make_node("Mul", [t3, half], [y], name=f"{prefix}_final"),
    ]
    return nodes, inits


def remove_safe_reshape_allowzero(model) -> int:
    """删除语义等价的 Reshape allowzero=1 属性.

    allowzero 只影响 shape 中值为 0 的维度.当前 Qualcomm ViT 的相关 shape
    均为非零常量,因此删除属性不会改变输出；遇到动态 shape 或包含 0 时拒绝猜测.
    """
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    removed = 0
    for node in model.graph.node:
        attributes = {attribute.name: attribute for attribute in node.attribute}
        allowzero = attributes.get("allowzero")
        if node.op_type != "Reshape" or allowzero is None:
            continue
        if allowzero.i == 0:
            node.attribute.remove(allowzero)
            removed += 1
            continue
        shape_tensor = initializers.get(node.input[1])
        if shape_tensor is None:
            raise ValueError(
                f"无法安全降级动态 Reshape allowzero=1: {node.name}")
        shape = numpy_helper.to_array(shape_tensor)
        if np.any(shape == 0):
            raise ValueError(
                f"无法安全降级含 0 shape 的 Reshape allowzero=1: "
                f"{node.name}, shape={shape.tolist()}")
        node.attribute.remove(allowzero)
        removed += 1
    return removed


def validate_target_schemas(model) -> None:
    """确认全部标准 ONNX 算子在目标 opset 中存在且属性受支持."""
    for node in model.graph.node:
        domain = node.domain or ""
        if domain not in {"", "ai.onnx"}:
            continue
        schema = onnx.defs.get_schema(node.op_type, TARGET_OPSET, domain)
        supported = set(schema.attributes)
        unknown = sorted(attribute.name for attribute in node.attribute
                         if attribute.name not in supported)
        if unknown:
            raise ValueError(
                f"{node.name or node.op_type} 在 opset {TARGET_OPSET} "
                f"存在不支持属性: {unknown}")


def downgrade(model_path: Path) -> None:
    """原地检查并执行降级, 完成后写回同一路径."""
    model = onnx.load(str(model_path))
    opsets = {(op.domain or "ai.onnx"): op.version
              for op in model.opset_import}
    gelu_count = sum(1 for node in model.graph.node
                     if node.op_type == "Gelu")
    needs = (model.ir_version > TARGET_IR
             or opsets.get("ai.onnx", 0) > MAX_SUPPORTED_OPSET
             or gelu_count > 0)
    print(f"[INFO] 当前 IR={model.ir_version}, opset={opsets}, "
          f"Gelu 节点={gelu_count}")
    if not needs:
        print("[INFO] 无需降级, 保持原样.")
        return

    reshape_count = remove_safe_reshape_allowzero(model)
    new_nodes = []
    new_inits = []
    gelu_index = 0
    for node in model.graph.node:
        if node.op_type == "Gelu":
            nodes, inits = expand_gelu(node, gelu_index)
            gelu_index += 1
            new_nodes.extend(nodes)
            new_inits.extend(inits)
        else:
            new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_inits)
    for op in model.opset_import:
        if (op.domain or "ai.onnx") == "ai.onnx" and op.version > TARGET_OPSET:
            op.version = TARGET_OPSET
    if model.ir_version > TARGET_IR:
        model.ir_version = TARGET_IR
    validate_target_schemas(model)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as error:  # noqa: BLE001 - 形状推断失败不阻断降级
        print(f"[WARN] 形状推断失败, 跳过: {error}")
    onnx.checker.check_model(model)
    onnx.save(model, str(model_path))
    print(f"[OK] 已降级: IR->{model.ir_version}, opset->{TARGET_OPSET}, "
          f"展开 Gelu x{gelu_index}, 移除 Reshape allowzero "
          f"x{reshape_count}, 节点总数 {len(model.graph.node)}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="待降级的单文件 ONNX, 原地写回.")
    return parser.parse_args()


if __name__ == "__main__":
    downgrade(parse_args().model)
