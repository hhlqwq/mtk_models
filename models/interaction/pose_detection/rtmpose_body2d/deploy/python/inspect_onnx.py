"""输出 RTMPose ONNX 中指定算子的上下游及静态形状."""

import argparse
from collections import Counter
from pathlib import Path

import onnx


def tensor_shapes(model: onnx.ModelProto) -> dict[str, list[int | str]]:
    """收集图输入、输出、中间张量和初始化器的形状."""
    shapes = {}
    values = list(model.graph.input) + list(model.graph.output)
    values.extend(model.graph.value_info)
    for value in values:
        dimensions = []
        for dimension in value.type.tensor_type.shape.dim:
            dimensions.append(
                dimension.dim_value if dimension.dim_value else
                dimension.dim_param or "?")
        shapes[value.name] = dimensions
    for tensor in model.graph.initializer:
        shapes[tensor.name] = list(tensor.dims)
    return shapes


def inspect_model(model_path: Path, operator: str) -> None:
    """打印目标算子及其直接生产者、消费者和张量形状."""
    model = onnx.load(str(model_path), load_external_data=False)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as error:  # noqa: BLE001 - 诊断脚本保留原图继续输出
        print(f"[WARN] shape inference 失败: {error}")
    shapes = tensor_shapes(model)
    producers = {
        output: (index, node) for index, node in enumerate(model.graph.node)
        for output in node.output
    }
    consumers = {}
    for index, node in enumerate(model.graph.node):
        for input_name in node.input:
            consumers.setdefault(input_name, []).append((index, node))
    counts = Counter(node.op_type for node in model.graph.node)
    print(f"[INFO] 节点类型统计: {dict(sorted(counts.items()))}")
    matches = [(index, node) for index, node in enumerate(model.graph.node)
               if node.op_type == operator]
    print(f"[INFO] {operator} 节点数量: {len(matches)}")
    for index, node in matches:
        print(f"[NODE] index={index}, name={node.name}")
        for input_name in node.input:
            producer = producers.get(input_name)
            producer_text = (
                f"{producer[0]}:{producer[1].op_type}:{producer[1].name}"
                if producer else "graph_input_or_initializer")
            print(f"  input={input_name}, shape={shapes.get(input_name)}, "
                  f"producer={producer_text}")
        for output_name in node.output:
            consumer_text = [
                f"{item[0]}:{item[1].op_type}:{item[1].name}"
                for item in consumers.get(output_name, [])
            ]
            print(f"  output={output_name}, shape={shapes.get(output_name)}, "
                  f"consumers={consumer_text}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--operator", default="GatherND")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    inspect_model(arguments.model, arguments.operator)
