"""从编译输入 TFLite 提取 ViT 板端精度测试所需的量化元数据."""

import argparse
import json
from pathlib import Path

import mtk_converter


def main() -> None:
    """检查输入输出张量,并保存可由板端直接读取的量化参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tflite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graph = mtk_converter.TFLiteParser(str(args.tflite))
    inputs = graph.get_input_tensor_details()
    outputs = graph.get_output_tensor_details()
    if (len(inputs) != 1 or list(inputs[0]["shape"]) != [1, 3, 224, 224] or
            len(outputs) != 1 or list(outputs[0]["shape"]) != [1, 1000]):
        raise ValueError("ViT TFLite 输入或输出结构异常.")
    values = {
        "input_scale": float(inputs[0]["quantization"]["scales"][0]),
        "input_zero_point": int(inputs[0]["quantization"]["zero_points"][0]),
        "output_scale": float(outputs[0]["quantization"]["scales"][0]),
        "output_zero_point": int(outputs[0]["quantization"]["zero_points"][0]),
        "input_shape": [1, 3, 224, 224],
        "output_shape": [1, 1000],
    }
    if values["input_scale"] <= 0 or values["output_scale"] <= 0:
        raise ValueError("量化 scale 必须为正值.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] ViT 量化元数据: {args.output}.")


if __name__ == "__main__":
    main()
