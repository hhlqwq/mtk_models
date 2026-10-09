"""转换 Google YAMNet 为默认 FP16 或第一折校准的可选 INT8 模型."""

import argparse
import csv
import json
from pathlib import Path
import sys

import mtk_converter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from audio_utils import load_records, load_waveform, preprocess


def tensor_metadata(detail):
    """验证单张量 INT8 类型和量化尺度,拒绝猜测默认尺度."""
    quantization = detail["quantization"]
    scales, zeros = quantization["scales"], quantization["zero_points"]
    dtype = str(detail.get("dtype", detail.get("type", ""))).lower()
    if len(scales) != 1 or float(scales[0]) <= 0 or len(zeros) != 1:
        raise ValueError(f"不支持的量化参数: {detail}.")
    if "int8" not in dtype or "uint8" in dtype:
        raise ValueError(f"IO 必须为 INT8: {detail}.")
    return {"shape": [int(value) for value in detail["shape"]],
            "scale": float(scales[0]), "zero_point": int(zeros[0])}


def main():
    """默认转换 FP16 权重,可选第一折每类两条音频的 200 窗 INT8 校准."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    args = parser.parse_args()
    if args.precision == "fp16":
        converter = mtk_converter.OnnxConverter.from_model_proto_file(
            str(args.models_dir / "model_fp32.onnx"))
        converter.convert_float32_weights_to_float16 = True
        converter.convert_to_tflite(str(args.models_dir / "model_fp16.tflite"))
        with (args.models_dir / "runtime_config.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([1, 0, "fp16"])
            writer.writerow([1, 0, "fp16"])
        (args.models_dir / "quantization.json").write_text(json.dumps({
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
        for path in tqdm(paths, desc="YAMNet INT8 校准"):
            for patch in preprocess(load_waveform(path))[:2]:
                yield [patch[None]]

    converter = mtk_converter.OnnxConverter.from_model_proto_file(
        str(args.models_dir / "model_fp32.onnx"))
    converter.quantize = True
    converter.use_per_output_channel_quantization = True
    converter.append_output_dequantize_ops = False
    converter.calibration_data_gen = calibration
    output = args.models_dir / "model_int8.tflite"
    converter.convert_to_tflite(str(output))
    reader = mtk_converter.TFLiteParser(str(output))
    inputs, outputs = reader.get_input_tensor_details(), reader.get_output_tensor_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("必须为单输入和单输出.")
    contract = [tensor_metadata(inputs[0]), tensor_metadata(outputs[0])]
    if contract[0]["shape"] != [1, 1, 96, 64] or contract[1]["shape"] != [1, 521]:
        raise ValueError(f"输入输出形状不匹配: {contract}.")
    with (args.models_dir / "runtime_config.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        for item in contract:
            writer.writerow([item["scale"], item["zero_point"]])
    (args.models_dir / "quantization.json").write_text(json.dumps({
        "precision": "int8",
        "input": contract[0], "output": contract[1], "calibration_fold": 1,
        "calibration_clips": [path.name for path in paths], "patches": 200,
        "converter": mtk_converter.__version__}, indent=2))
    print("[OK] INT8 模型与运行参数已生成.", flush=True)


if __name__ == "__main__":
    main()
