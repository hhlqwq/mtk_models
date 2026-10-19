"""使用原有 W4A16 协议量化,严格核对实际输出分片."""

import json

from tools.deploy import arguments
from tools.llm.native import python, settings


def main():
    """复用已完成的相同批次校准,量化成功后记录真实设置."""
    native = settings(arguments(__file__))
    if (native.work / "calibration_complete.txt").read_text().strip() != str(native.batches):
        raise ValueError("校准尚未完成或批次数不一致.")
    (native.work / "quantization_settings.json").unlink(missing_ok=True)
    extra = []
    if native.optimization != "none":
        extra += ["--weight_optimization", native.optimization]
    if native.extra:
        extra += ["--extra_converter_options", native.extra]
    python(native, native.ctx.repo / "tools/llm/quantize_native.py", "converter",
           native.model / "config.json", "-p", "asym4W_sym16A", "-tp", "asym4W_sym16A",
           "-d", native.work / "calibration_datasets" / native.model.name, "-m", "Overall", *extra)
    config = json.loads((native.model / "config.json").read_text())
    expected = {f"{native.prefix}_{index}.tflite" for index in range(config["num_hidden_layers"] + 1)}
    if {path.name for path in native.quantized.glob("*.tflite")} != expected:
        raise ValueError("SDK 量化分片不完整.")
    for name in expected:
        path = native.quantized / name
        if not path.stat().st_size or not path.with_name(path.stem + "_info.json").is_file():
            raise ValueError(f"分片为空或缺少 SDK 元数据: {name}.")
    (native.work / "quantization_settings.json").write_text(json.dumps({
        "attention": "asym4W_sym16A", "mlp": "asym4W_sym16A", "lm_head": "asym4W_sym16A",
        "quantized_prefix": native.prefix}, indent=2) + "\n")


if __name__ == "__main__":
    main()
