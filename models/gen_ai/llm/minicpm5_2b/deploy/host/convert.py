"""顺序执行 MiniCPM5 校准、W4A16 量化和双静态图导出."""

import json
import os
import shlex

from tools.deploy import arguments
from tools.files import single_file
from tools.llm.native import settings
from tools.llm.quantize_native import fix_native_shapes, quantize
from tools.process import run_command


def calibrate(native):
    """直接执行 SDK 校准命令,成功后记录实际完成的批次数."""
    marker = native.work / "calibration_complete.txt"
    marker.unlink(missing_ok=True)
    command = [
        native.python, native.python.parent / "mtk_make_llm_ptq_calib_dataset",
        "converter", native.model / "config.json",
        native.work / "calibration.jsonl", "-b", str(native.batches), "-m", "128",
    ]
    run_command(command, cwd=native.work)
    marker.write_text(f"{native.batches}\n")


def quantize_model(native):
    """在当前 Python 进程调用 SDK 适配函数并检查全部量化分片."""
    marker = native.work / "calibration_complete.txt"
    if marker.read_text().strip() != str(native.batches):
        raise ValueError("校准尚未完成或批次数不一致.")
    record = native.work / "quantization_settings.json"
    record.unlink(missing_ok=True)
    options = [
        "converter", str(native.model / "config.json"),
        "-p", "asym4W_sym16A", "-tp", "asym4W_sym16A",
        "-d", str(native.work / "calibration_datasets" / native.model.name),
        "-m", "Overall",
    ]
    if native.optimization != "none":
        options.extend(["--weight_optimization", native.optimization])
    if native.extra:
        options.extend(["--extra_converter_options", native.extra])
    quantize(options)
    config = json.loads((native.model / "config.json").read_text())
    expected = {
        f"{native.prefix}_{index}.tflite"
        for index in range(config["num_hidden_layers"] + 1)
    }
    if {path.name for path in native.quantized.glob("*.tflite")} != expected:
        raise ValueError("SDK 量化分片不完整.")
    for name in expected:
        path = native.quantized / name
        if not path.stat().st_size or not path.with_name(path.stem + "_info.json").is_file():
            raise ValueError(f"分片为空或缺少 SDK 元数据: {name}.")
    record.write_text(json.dumps({
        "attention": "asym4W_sym16A", "mlp": "asym4W_sym16A",
        "lm_head": "asym4W_sym16A", "quantized_prefix": native.prefix,
        "weight_optimization": native.optimization,
    }, indent=2) + "\n")


def export_shapes(native):
    """直接导出两张静态图,将实际编译路径交给 Shell 入口."""
    shapes = [f"{native.prefill}t{native.context}c", f"1t{native.context}c"]
    fix_native_shapes(native.quantized, shapes)
    sources = [
        single_file(native.quantized.with_name(f"{native.prefix}_{shape}"), "*.tflite")
        for shape in shapes
    ]
    # 此文件只包含转义后的本次模型路径,不包含主机或连接信息.
    values = dict(zip(("PREFILL_TFLITE", "DECODE_TFLITE"), sources))
    (native.work / "compile.env").write_text("".join(
        f"{name}={shlex.quote(str(path))}\n" for name, path in values.items()))
    print(f"[编译输入] {native.work / 'compile.env'}", flush=True)


def main():
    """在固定工作目录执行转换函数,不再启动仓库内的 Python 子脚本."""
    native = settings(arguments(__file__))
    (native.work / "compile.env").unlink(missing_ok=True)
    previous = os.getcwd()
    try:
        os.chdir(native.work)
        print("[转换 1/3] 生成校准数据.", flush=True)
        calibrate(native)
        print("[转换 2/3] 全 W4A16 量化并核对实际分片.", flush=True)
        quantize_model(native)
        print("[转换 3/3] 导出 Prefill 和 Decode 静态图.", flush=True)
        export_shapes(native)
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    main()
