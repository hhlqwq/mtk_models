#!/usr/bin/env python3
"""将 YOLO-World XL 固定词表 ONNX 改写为 Genio 720 纯 Neuron 图."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def parse_args() -> argparse.Namespace:
    """解析原始 Raw ONNX 与纯 NPU ONNX 路径."""
    parser = argparse.ArgumentParser(description="改写 YOLO-World XL 固定词表 ONNX.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def add_shape(graph: onnx.GraphProto, name: str, values: list[int]) -> str:
    """为静态 Reshape 添加整型形状常量."""
    graph.initializer.append(
        numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)
    )
    return name


def prune_unused_nodes(model: onnx.ModelProto) -> None:
    """删除被等价卷积替代后不可达的旧分支和权重."""
    producers = {
        output: node for node in model.graph.node for output in node.output
    }
    reachable: set[str] = set()
    pending = [output.name for output in model.graph.output]
    while pending:
        node = producers.get(pending.pop())
        if node is None or node.name in reachable:
            continue
        reachable.add(node.name)
        pending.extend(node.input)

    nodes = [node for node in model.graph.node if node.name in reachable]
    used_inputs = {name for node in nodes for name in node.input}
    initializers = [
        value for value in model.graph.initializer if value.name in used_inputs
    ]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)


def rewrite_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """以卷积、切片和池化替换会回退或计算错误的 ONNX 算子."""
    graph = model.graph
    original_nodes = list(graph.node)
    producers = {
        output: node for node in original_nodes for output in node.output
    }
    initializers = {
        value.name: numpy_helper.to_array(value) for value in graph.initializer
    }
    shapes = {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in graph.value_info
    }
    text_node = producers[
        "/baseModel/neck/top_down_layers.0/attn_block/guide_fc/Constant_output_0"
    ]
    text = numpy_helper.to_array(
        next(attribute.t for attribute in text_node.attribute if attribute.name == "value")
    )
    class_node = producers["onnx::MatMul_1551"]
    class_matrix = numpy_helper.to_array(
        next(attribute.t for attribute in class_node.attribute if attribute.name == "value")
    )
    if text.shape != (1, 80, 512) or class_matrix.shape != (1, 512, 80):
        raise ValueError("文本嵌入或分类矩阵形状不匹配,不能改写.")
    class_weight = np.ascontiguousarray(class_matrix[0].T.reshape(80, 512, 1, 1))
    dfl_node = producers["onnx::MatMul_1582"]
    dfl_bins = numpy_helper.to_array(
        next(attribute.t for attribute in dfl_node.attribute if attribute.name == "value")
    )
    if dfl_bins.shape != (16, 1):
        raise ValueError("DFL 加权矩阵形状不匹配,不能改写.")
    dfl_weight = np.ascontiguousarray(
        np.tile(dfl_bins.reshape(1, 16, 1, 1), (4, 1, 1, 1))
    )
    dfl_outputs = {
        f"/baseModel/head_module/MatMul{suffix}": producers[
            f"/baseModel/head_module/Reshape{output_suffix}_output_0"
        ]
        for suffix, output_suffix in ()
    }
    dfl_outputs = {
        suffix: next(
            node.output[0]
            for node in original_nodes
            if node.name == f"/baseModel/head_module/Reshape_{2 * index + 1}"
        )
        for index, suffix in enumerate(("", "_1", "_2"))
    }

    attention: dict[str, tuple[int, int, int, int, int]] = {}
    for node in original_nodes:
        if node.op_type != "MatMul" or "/attn_block/MatMul" not in node.name:
            continue
        prefix = node.name.rsplit("/MatMul", 1)[0]
        left = producers[node.input[0]]
        transpose = producers[left.input[0]]
        reshape = producers[transpose.input[0]]
        features = reshape.input[0]
        _, channels, height, width = shapes[features]
        _, heads, depth, classes = shapes[node.input[1]]
        if classes != 80 or channels != heads * depth or height != width:
            raise ValueError(f"注意力维度不匹配: {prefix}.")

        guide = producers[prefix + "/guide_fc/MatMul_output_0"]
        guide_weight = initializers[guide.input[1]]
        bias_name = (
            "baseModel.neck."
            + prefix.split("/neck/", 1)[1].replace("/", ".")
            + ".guide_fc.bias"
        )
        guide_bias = initializers[bias_name]
        projected = (text @ guide_weight + guide_bias).reshape(
            1, classes, heads, depth
        )
        weights = np.ascontiguousarray(
            projected.transpose(0, 2, 3, 1)[0]
            .transpose(0, 2, 1)
            .reshape(heads * classes, depth, 1, 1)
        )
        key = prefix.replace("/", "_")
        weight_name = key + "_groupconv_weight"
        graph.initializer.append(numpy_helper.from_array(weights, name=weight_name))
        attention[prefix] = (heads, depth, classes, height, width)
        attention[prefix + "/features"] = features

    if len(attention) != 8:
        raise ValueError("期望四个固定文本注意力模块.")

    new_nodes = []
    counts = {
        "attention": 0, "pool": 0, "repeat": 0, "split": 0,
        "class": 0, "dfl": 0,
    }
    for node in original_nodes:
        name = node.name
        if node.op_type == "MatMul" and "/attn_block/MatMul" in name:
            prefix = name.rsplit("/MatMul", 1)[0]
            heads, _, _, _, _ = attention[prefix]
            key = prefix.replace("/", "_")
            new_nodes.append(
                helper.make_node(
                    "Conv",
                    [attention[prefix + "/features"], key + "_groupconv_weight"],
                    [key + "_conv"],
                    name=key + "_groupconv",
                    group=heads,
                    kernel_shape=[1, 1],
                )
            )
            counts["attention"] += 1
        elif node.op_type == "ReduceMax" and "/attn_block/" in name:
            prefix = name.rsplit("/ReduceMax", 1)[0]
            heads, _, classes, height, width = attention[prefix]
            positions = height * width
            key = prefix.replace("/", "_")
            shape0 = add_shape(
                graph, key + "_pool_shape0", [1, heads, classes, positions]
            )
            shape1 = add_shape(
                graph, key + "_pool_shape1", [1, heads * positions, 1, classes]
            )
            shape2 = add_shape(
                graph, key + "_pool_shape2", [1, heads, height, width]
            )
            new_nodes.extend(
                [
                    helper.make_node(
                        "Reshape", [key + "_conv", shape0], [key + "_pool_r0"],
                        name=key + "_pool_r0",
                    ),
                    helper.make_node(
                        "Transpose", [key + "_pool_r0"], [key + "_pool_t"],
                        name=key + "_pool_t", perm=[0, 1, 3, 2],
                    ),
                    helper.make_node(
                        "Reshape", [key + "_pool_t", shape1], [key + "_pool_r1"],
                        name=key + "_pool_r1",
                    ),
                    helper.make_node(
                        "MaxPool", [key + "_pool_r1"], [key + "_pool_m1"],
                        name=key + "_pool_m1", kernel_shape=[1, 5], strides=[1, 5],
                    ),
                    helper.make_node(
                        "MaxPool", [key + "_pool_m1"], [key + "_pool_m2"],
                        name=key + "_pool_m2", kernel_shape=[1, 4], strides=[1, 4],
                    ),
                    helper.make_node(
                        "MaxPool", [key + "_pool_m2"], [key + "_pool_m3"],
                        name=key + "_pool_m3", kernel_shape=[1, 4], strides=[1, 4],
                    ),
                    helper.make_node(
                        "Reshape", [key + "_pool_m3", shape2], list(node.output),
                        name=key + "_pool_out",
                    ),
                ]
            )
            counts["pool"] += 1
        elif name.endswith("/Reshape_5") and "/attn_block/" in name:
            prefix = name.rsplit("/Reshape_5", 1)[0]
            heads, depth, _, _, _ = attention[prefix]
            key = prefix.replace("/", "_")
            repeat_weight = np.zeros((heads * depth, heads, 1, 1), dtype=np.float32)
            for channel in range(heads * depth):
                repeat_weight[channel, channel // depth, 0, 0] = 1.0
            weight_name = key + "_repeat_weight"
            graph.initializer.append(
                numpy_helper.from_array(repeat_weight, name=weight_name)
            )
            repeated = key + "_repeat_output"
            new_nodes.extend(
                [
                    helper.make_node(
                        "Conv", [prefix + "/Mul_output_0", weight_name],
                        [repeated], name=key + "_repeat", kernel_shape=[1, 1],
                    ),
                    helper.make_node(
                        "Mul", [prefix + "/project_conv/conv/Conv_output_0", repeated],
                        list(node.output), name=key + "_mul4d",
                    ),
                ]
            )
            counts["repeat"] += 1
        elif node.op_type == "MatMul" and name.startswith(
            "/baseModel/head_module/MatMul"
        ):
            suffix = name.removeprefix("/baseModel/head_module/MatMul")
            if suffix not in dfl_outputs:
                raise ValueError(f"未知 DFL 分支: {name}.")
            _, positions, directions, bins = shapes[node.input[0]]
            height = width = int(positions**0.5)
            if directions != 4 or bins != 16 or height * width != positions:
                raise ValueError(f"DFL 张量形状不匹配: {name}.")
            key = "codex_dfl" + suffix
            shape_name = add_shape(graph, key + "_shape", [1, 64, height, width])
            weight_name = key + "_weight"
            graph.initializer.append(
                numpy_helper.from_array(dfl_weight, name=weight_name)
            )
            new_nodes.extend(
                [
                    helper.make_node(
                        "Transpose", [node.input[0]], [key + "_transposed"],
                        name=key + "_transpose", perm=[0, 2, 3, 1],
                    ),
                    helper.make_node(
                        "Reshape", [key + "_transposed", shape_name],
                        [key + "_reshaped"], name=key + "_reshape",
                    ),
                    helper.make_node(
                        "Conv", [key + "_reshaped", weight_name],
                        [dfl_outputs[suffix]], name=key + "_conv", group=4,
                        kernel_shape=[1, 1],
                    ),
                ]
            )
            counts["dfl"] += 1
        elif name in {
            "/baseModel/head_module/Reshape_1",
            "/baseModel/head_module/Reshape_3",
            "/baseModel/head_module/Reshape_5",
        }:
            continue
        elif node.op_type == "Split" and (
            "/image_model/" in name or "/neck/" in name
        ):
            attributes = {
                item.name: helper.get_attribute_value(item) for item in node.attribute
            }
            sizes = attributes.get("split")
            if sizes is None and len(node.input) == 2:
                split_node = producers[node.input[1]]
                sizes = numpy_helper.to_array(
                    next(
                        item.t
                        for item in split_node.attribute
                        if item.name == "value"
                    )
                ).tolist()
            if attributes.get("axis") != 1 or sizes is None or len(sizes) != 2:
                raise ValueError(f"无法转换 Split: {name}.")
            key = "codex_split_" + str(counts["split"])
            for index, (start, end) in enumerate(
                ((0, sizes[0]), (sizes[0], sum(sizes)))
            ):
                starts = add_shape(graph, key + f"_start{index}", [start])
                ends = add_shape(graph, key + f"_end{index}", [end])
                axes = add_shape(graph, key + f"_axis{index}", [1])
                new_nodes.append(
                    helper.make_node(
                        "Slice", [node.input[0], starts, ends, axes],
                        [node.output[index]], name=key + f"_{index}",
                    )
                )
            counts["split"] += 1
        else:
            branch = next(
                (str(index) for index in range(3) if f"/cls_contrasts.{index}/" in name),
                None,
            )
            if branch is not None and name.endswith("/Transpose"):
                weight_name = f"codex_cls_contrasts.{branch}.conv_weight"
                graph.initializer.append(
                    numpy_helper.from_array(class_weight, name=weight_name)
                )
                new_nodes.append(
                    helper.make_node(
                        "Conv", [node.input[0], weight_name],
                        [f"/baseModel/head_module/cls_contrasts.{branch}/Transpose_1_output_0"],
                        name=f"codex_cls_contrasts.{branch}.Conv",
                        kernel_shape=[1, 1],
                    )
                )
                counts["class"] += 1
            elif branch is not None and any(
                name.endswith("/" + suffix)
                for suffix in (
                    "Constant", "Reshape", "MatMul", "Constant_1",
                    "Reshape_1", "Transpose_1",
                )
            ):
                continue
            else:
                new_nodes.append(node)

    expected = {
        "attention": 4, "pool": 4, "repeat": 4, "split": 8,
        "class": 3, "dfl": 3,
    }
    if counts != expected:
        raise ValueError(f"算子数量不匹配: {counts}.")
    del graph.node[:]
    graph.node.extend(new_nodes)
    prune_unused_nodes(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    """生成并校验仍以 ONNX 保存的纯 Neuron 模型."""
    args = parse_args()
    print(f"[1/3] 读取 ONNX: {args.input}")
    model = onnx.load(args.input)
    print("[2/3] 等价替换固定词表矩阵运算和 CPU 回退算子.")
    model = rewrite_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"[3/3] 已保存纯 Neuron 候选 ONNX: {args.output}")


if __name__ == "__main__":
    main()
