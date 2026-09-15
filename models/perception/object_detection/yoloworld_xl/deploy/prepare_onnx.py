#!/usr/bin/env python3
"""为 Genio Neuron EP 生成等价的 YOLO-World opset 13 模型。"""

import argparse
import hashlib
from pathlib import Path

import onnx
from onnx import version_converter


EXPECTED_INPUT_SHAPE = [1, 3, 640, 640]
EXPECTED_OUTPUT_SHAPES = [
    [1, 80, 80, 80],
    [1, 4, 80, 80],
    [1, 80, 40, 40],
    [1, 4, 40, 40],
    [1, 80, 20, 20],
    [1, 4, 20, 20],
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="将官方 YOLO-World opset 11 模型等价转换为 opset 13。"
    )
    parser.add_argument("--input", type=Path, required=True, help="官方 ONNX 路径。")
    parser.add_argument("--output", type=Path, required=True, help="兼容 ONNX 路径。")
    return parser.parse_args()


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int]:
    """读取静态张量形状。"""
    return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_model(model: onnx.ModelProto) -> None:
    """检查官方模型接口和待修复的 Softmax 条件。"""
    opsets = {item.domain: item.version for item in model.opset_import}
    if opsets.get("") != 11:
        raise ValueError(f"期望默认域 opset 11，实际为 {opsets.get('')}。")
    if len(model.graph.input) != 1:
        raise ValueError("模型必须只有一个图像输入。")
    if tensor_shape(model.graph.input[0]) != EXPECTED_INPUT_SHAPE:
        raise ValueError(f"输入形状不匹配: {tensor_shape(model.graph.input[0])}。")
    output_shapes = [tensor_shape(item) for item in model.graph.output]
    if output_shapes != EXPECTED_OUTPUT_SHAPES:
        raise ValueError(f"输出形状不匹配: {output_shapes}。")

    softmax_nodes = [node for node in model.graph.node if node.op_type == "Softmax"]
    if len(softmax_nodes) != 3:
        raise ValueError(f"期望 3 个 Softmax，实际为 {len(softmax_nodes)}。")
    for node in softmax_nodes:
        attributes = {
            item.name: onnx.helper.get_attribute_value(item) for item in node.attribute
        }
        if attributes.get("axis") != 3:
            raise ValueError(f"Softmax {node.name} 的 axis 不是 3。")


def main() -> None:
    """转换模型并执行 ONNX 结构校验。"""
    args = parse_args()
    print(f"[1/4] 读取官方模型: {args.input}")
    model = onnx.load(args.input)
    validate_source_model(model)

    print("[2/4] 使用 ONNX version converter 转换到 opset 13。")
    converted = version_converter.convert_version(model, 13)

    print("[3/4] 执行 ONNX checker。")
    onnx.checker.check_model(converted, full_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, args.output)

    print("[4/4] 输出模型校验值。")
    print(f"source_sha256={sha256_file(args.input)}")
    print(f"output_sha256={sha256_file(args.output)}")
    print(f"output_size={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
