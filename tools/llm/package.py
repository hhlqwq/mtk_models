"""读取 LLM 原生部署包使用的公共张量契约."""

import math
import shutil

import numpy as np

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
            quantization = tensor.Quantization()
            if quantization is None or quantization.ScaleLength() != 1 or quantization.ZeroPointLength() != 1:
                raise ValueError("原生 I/O 需要逐张量 INT16 量化参数.")
            result[kind].append({"name": tensor.Name().decode(), "dtype": "INT16",
                                 "shape": tensor.ShapeAsNumpy().tolist(),
                                 "scale": float(quantization.Scale(0)),
                                 "zero_point": int(quantization.ZeroPoint(0))})
    return result


def generation_resources(model, output, contracts):
    """在打包时保存 SDK 掩码与 RoPE,板端无需安装转换 SDK 或重新编译."""
    from mtk_llm_sdk.utils import generate_utils, utils

    config = utils.resolve_model_classes(str(model / "config.json"), bypass_tokenizer=True)[0]
    context = config.max_position_embeddings
    master = generate_utils.get_master_rot_emb(config, np.float32)
    arrays = {}
    for phase, contract in contracts.items():
        print(f"[生成输入] {phase}: SDK mask 与 RoPE.", flush=True)
        batch = contract["inputs"][0]["shape"][1]
        if context % batch:
            raise ValueError("静态上下文必须是 Prefill 批长的整数倍.")
        masks, positions = [], []
        for offset in range(0, context, batch):
            values = {"mask": generate_utils.generate_mask(context, offset, batch, batch,
                mask_value=config.mask_value, dtype=np.float32),
                "pos_emb": master[:, :, offset:offset + batch, :]}
            for name, bank in (("mask", masks), ("pos_emb", positions)):
                item = next(row for row in contract["inputs"] if row["name"] == name)
                data = values[name] / item["scale"] + item["zero_point"]
                if list(data.shape) != item["shape"] or not np.isfinite(data).all() or np.any(
                        (data < -32768.5) | (data > 32767.5)):
                    raise ValueError(f"SDK 生成输入与量化契约不符: {phase}/{name}.")
                bank.append(np.clip(np.rint(data), -32768, 32767).astype("<i2"))
        arrays[f"{phase}_mask"] = np.stack(masks)
        arrays[f"{phase}_pos_emb"] = np.stack(positions)
    np.savez(output / "tokenizer/generation_inputs.npz", **arrays)
    shutil.copyfile(model / "tokenizer.json", output / "tokenizer/tokenizer.json")
