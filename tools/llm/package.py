"""LLM 原生部署包的公共张量契约与入口文件准备."""

import math
from pathlib import Path
import shutil

from mtk_converter.python.converters.tflite.schema.tflite.Model import Model
from mtk_converter.python.converters.tflite.schema.tflite.TensorType import TensorType


def graph_contract(path):
    """读取原生静态图 I/O,要求配置声明的所有接口实际为 INT16."""

    graph = Model.GetRootAsModel(path.read_bytes(), 0).Subgraphs(0)
    result = {"inputs": [], "outputs": []}
    checked = 0
    for index in range(graph.TensorsLength()):
        tensor = graph.Tensors(index)
        quantization = tensor.Quantization()
        if quantization is not None:
            for offset in range(quantization.ScaleLength()):
                scale = quantization.Scale(offset)
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError(f"量化 Scale 非法: {tensor.Name()},scale={scale}")
                checked += 1
    result["quantization_scales_checked"] = checked
    for kind, length, index in (("inputs", graph.InputsLength, graph.Inputs),
                                ("outputs", graph.OutputsLength, graph.Outputs)):
        for offset in range(length()):
            tensor = graph.Tensors(index(offset))
            if tensor.Type() != TensorType.INT16:
                raise ValueError(f"原生图包含非 INT16 的 {kind}: {tensor.Name()}")
            result[kind].append({"name": tensor.Name().decode(), "dtype": "INT16",
                                 "shape": tensor.ShapeAsNumpy().tolist()})
    return result


def write_compile_scripts(output):
    """复制共享原生 CLI 入口,保留原有 Token 计数与日志规则."""
    for source, destination in (("compile_native.sh", "compile.sh"),
                                ("evaluate_native.sh", "evaluate.sh")):
        shutil.copyfile(Path(__file__).with_name(source), output / destination)
