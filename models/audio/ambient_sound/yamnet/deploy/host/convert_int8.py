"""转换 Google YAMNet 为默认 FP16 或第一折校准的 INT8/W8A16 模型."""

import argparse
import csv
import json
from pathlib import Path
import sys

import mtk_converter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from audio_utils import load_records, load_waveform, preprocess


def tensor_metadata(detail, dtype_name):
    """验证单张量真实整数类型和量化尺度,拒绝猜测默认尺度."""
    quantization = detail["quantization"]
    scales, zeros = quantization["scales"], quantization["zero_points"]
    dtype = str(detail.get("dtype", detail.get("type", ""))).lower()
    if len(scales) != 1 or float(scales[0]) <= 0 or len(zeros) != 1:
        raise ValueError(f"不支持的量化参数: {detail}.")
    if dtype_name not in dtype or f"u{dtype_name}" in dtype:
        raise ValueError(f"IO 必须为 {dtype_name}: {detail}.")
    return {"shape": [int(value) for value in detail["shape"]],
            "dtype": dtype_name, "scale": float(scales[0]), "zero_point": int(zeros[0])}


def audit_w8a16(reader):
    """检查每个主计算层的真实 W8A16 类型,同时记录偏置精度."""
    graph = reader.as_dict()
    subgraph = graph["subgraphs"][0]
    tensors = subgraph["tensors"]
    layers = []
    for operator in subgraph["operators"]:
        code = graph["operator_codes"][operator["opcode_index"]]
        name = code.get("custom_code", code["builtin_code"])
        if name not in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
                        "MTKEXT_CONV_2D", "MTKEXT_DEPTHWISE_CONV_2D",
                        "MTKEXT_FULLY_CONNECTED"):
            continue
        activation, weights, bias = [tensors[index] for index in operator["inputs"][:3]]
        output = tensors[operator["outputs"][0]]
        if (activation["type"], weights["type"], output["type"]) != (
                "INT16", "INT8", "INT16"):
            raise ValueError(f"主计算层不是 W8A16: {name}, {activation}, {weights}, {output}.")
        layers.append({"operator": name, "input_dtype": activation["type"],
                       "weight_dtype": weights["type"], "bias_dtype": bias["type"],
                       "output_dtype": output["type"]})
    if len(layers) != 28:
        raise ValueError(f"YAMNet 应有 27 层卷积与 1 层全连接,实际为 {len(layers)}.")
    return {"verified_affine_layers": len(layers), "layers": layers}


def main():
    """默认转换 FP16 权重,可选第一折每类两条音频的 200 窗 INT8 校准."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp16", "int8", "w8a16"], default="fp16")
    args = parser.parse_args()
    if args.precision == "fp16":
        converter = mtk_converter.OnnxConverter.from_model_proto_file(
            str(args.models_dir / "model_fp32.onnx"))
        converter.convert_float32_weights_to_float16 = True
        converter.convert_to_tflite(str(args.models_dir / "model_fp16.tflite"))
        with (args.models_dir / "runtime_config_fp16.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([1, 0, "fp16"])
            writer.writerow([1, 0, "fp16"])
        (args.models_dir / "quantization_fp16.json").write_text(json.dumps({
            "precision": "fp16", "converter": mtk_converter.__version__,
            "weights": "FP16", "ncc_flags": ["--relax-fp32", "--suppress-input",
                                               "--suppress-output", "--disallow-bridge"],
            "native_input_shape": [1, 1, 96, 64], "native_output_shape": [1, 521],
            "calibration_patches": 0}, indent=2))
        print("[OK] FP16 权重模型与原生 IO 参数已生成,需 NCC 降精度编译.", flush=True)
        return
    records = load_records(args.dataset)
    paths = []
    for target in range(50):
        chosen = [row for row in records if int(row["fold"]) == 1
                  and int(row["target"]) == target][:2]
        if len(chosen) != 2:
            raise ValueError("校准折必须每类至少两条音频.")
        paths.extend(args.dataset / "audio" / row["filename"] for row in chosen)

    def calibration():
        """按固定顺序提供 200 个真实音频特征窗口."""
        for path in tqdm(paths, desc=f"YAMNet {args.precision} 校准"):
            for patch in preprocess(load_waveform(path))[:2]:
                yield [patch[None]]

    converter = mtk_converter.OnnxConverter.from_model_proto_file(
        str(args.models_dir / "model_fp32.onnx"))
    converter.quantize = True
    converter.use_per_output_channel_quantization = True
    if args.precision == "w8a16":
        converter.allow_asym16_quantization = True
        converter.default_quantization_bitwidth = 16
        converter.default_weights_quantization_bitwidth = 8
        converter.input_quantization_bitwidths = 16
    converter.append_output_dequantize_ops = False
    converter.calibration_data_gen = calibration
    output = args.models_dir / f"model_{args.precision}.tflite"
    converter.convert_to_tflite(str(output))
    reader = mtk_converter.TFLiteParser(str(output))
    inputs, outputs = reader.get_input_tensor_details(), reader.get_output_tensor_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("必须为单输入和单输出.")
    dtype_name = "int16" if args.precision == "w8a16" else "int8"
    contract = [tensor_metadata(inputs[0], dtype_name), tensor_metadata(outputs[0], dtype_name)]
    audit = audit_w8a16(reader) if args.precision == "w8a16" else None
    if contract[0]["shape"] != [1, 1, 96, 64] or contract[1]["shape"] != [1, 521]:
        raise ValueError(f"输入输出形状不匹配: {contract}.")
    with (args.models_dir / f"runtime_config_{args.precision}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        for item in contract:
            row = [item["scale"], item["zero_point"]]
            if args.precision == "w8a16":
                row.append("w8a16")
            writer.writerow(row)
    (args.models_dir / f"quantization_{args.precision}.json").write_text(json.dumps({
        "precision": args.precision, "precision_audit": audit,
        "input": contract[0], "output": contract[1], "calibration_fold": 1,
        "calibration_clips": [path.name for path in paths], "patches": 200,
        "converter": mtk_converter.__version__}, indent=2))
    print(f"[OK] {args.precision} 模型与独立运行参数已生成.", flush=True)


if __name__ == "__main__":
    main()
