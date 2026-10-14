"""通过 MTK ONNX Converter 离线校准 FastSAM-s 并记录张量契约."""

import argparse
import sys
import json
import subprocess
from pathlib import Path


def tensor_metadata(detail):
    """验证逐张量 INT8 量化参数并转换为可序列化元数据."""
    quantization = detail["quantization"]
    scales, zeros = quantization["scales"], quantization["zero_points"]
    if len(scales) != 1 or len(zeros) != 1 or not float(scales[0]) > 0:
        raise ValueError(f"不支持的 IO 量化参数: {detail}.")
    # 数据类型必须由解析器明确提供,不能把浮点输出猜成 INT8.
    dtype = detail.get("dtype", detail.get("type"))
    if dtype is None or "int8" not in str(dtype).lower() or "uint8" in str(dtype).lower():
        raise ValueError(f"要求 INT8 IO,请核实 TFLiteParser 类型字段: {detail}.")
    return {"name": str(detail["name"]),
            "shape": [int(value) for value in detail["shape"]],
            "dtype": "int8", "scale": float(scales[0]),
            "zero_point": int(zeros[0])}


def main():
    """选取固定校准集,执行转换,按唯一形状映射运行时输出."""
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--config", type=Path, help="使用带备注的 YAML 配置.")
    parser.add_argument("--work-dir", type=Path, help="复用指定的中间文件目录.")
    parser.add_argument("--inside-container", action="store_true", help=argparse.SUPPRESS)
    options, _ = parser.parse_known_args()
    parser.add_argument("-h", "--help", action="help", help="显示参数说明.")
    paths = None
    samples = 100
    if options.config:
        from datetime import date
        import os
        import yaml
        sys.path.insert(0, str(Path(__file__).resolve().parents[6] / "tools"))
        from storage import load_model_config
        config = load_model_config(options.config)
        model = Path(__file__).resolve().parents[2]
        dataset = Path(config["data"]["dataset_dir"])
        work = options.work_dir or Path(config["build"]["work_root"]) / os.environ.get("FASTSAM_RUN_DATE", date.today().isoformat()) / "fastsam"
        work = work.resolve()
        for name in ("tmp", "cache"):
            (work / name).mkdir(parents=True, exist_ok=True)
        os.environ.update(PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(work / "tmp"),
                          XDG_CACHE_HOME=str(work / "cache"), TORCH_HOME=str(work / "cache/torch"))
        paths = {"output": model / config["model"]["output_dir"],
                 "calibration": dataset / config["data"]["calibration_dir"]}
        samples = config["data"]["calibration_samples"]
    parser.add_argument("--onnx", type=Path, required=paths is None,
                        default=paths["output"] / "model_fp32.onnx" if paths else None)
    parser.add_argument("--calibration-dir", type=Path, required=paths is None,
                        default=paths["calibration"] if paths else None)
    parser.add_argument("--samples", type=int, default=samples)
    parser.add_argument("--output", type=Path, required=paths is None,
                        default=paths["output"] / "model_int8.tflite" if paths else None)
    args = parser.parse_args()
    if options.config and not args.inside_container and not Path("/.dockerenv").exists():
        container = config["build"]["container"]
        subprocess.run(["docker", "exec", "-e", "MTK_MODEL_STORAGE_ROOT", "-e", "MTK_DATASET_ROOT", container, "mkdir", "-p", str(work / "tmp"), str(work / "cache")], check=True)
        print(f"[量化] 在 Docker {container} 中执行.", flush=True)
        subprocess.run(["docker", "exec", "-e", "MTK_MODEL_STORAGE_ROOT", "-e", "MTK_DATASET_ROOT", "-e", "PYTHONDONTWRITEBYTECODE=1",
                        "-e", f"TMPDIR={work / 'tmp'}", "-e", f"XDG_CACHE_HOME={work / 'cache'}",
                        "-e", f"TORCH_HOME={work / 'cache/torch'}", "-w", str(work), container,
                        "python", str(Path(__file__).resolve()), "--inside-container",
                        "--config", str(args.config.resolve()), "--work-dir", str(work),
                        "--onnx", str(args.onnx.resolve()), "--calibration-dir", str(args.calibration_dir.resolve()),
                        "--samples", str(args.samples), "--output", str(args.output.resolve())], check=True)
        return
    # Converter 只在容器内加载,预处理继续复用板端实现.
    import cv2
    import mtk_converter
    from tqdm import tqdm
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
    from fastsam_utils import OUTPUT_NAMES, OUTPUT_SHAPES, preprocess
    paths = sorted(path for path in args.calibration_dir.iterdir()
                   if path.suffix.lower() in {".jpg", ".jpeg", ".png"})[:args.samples]
    if args.samples <= 0 or len(paths) != args.samples:
        raise ValueError("校准图片不足或 samples 非正数.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibration = [{"path": str(path)} for path in paths]

    def calibration_data():
        """按固定顺序提供与板端一致的预处理输入."""
        for path in tqdm(paths, desc="FastSAM 校准"):
            yield [preprocess(cv2.imread(str(path)))[0]]

    print("[CONVERT] 开始 INT8 校准和转换.", flush=True)
    converter = mtk_converter.OnnxConverter.from_model_proto_file(str(args.onnx))
    converter.quantize = True
    converter.use_per_output_channel_quantization = True
    converter.append_output_dequantize_ops = False
    converter.calibration_data_gen = calibration_data
    converter.convert_to_tflite(str(args.output))
    reader = mtk_converter.TFLiteParser(str(args.output))
    inputs = reader.get_input_tensor_details()
    if len(inputs) != 1:
        raise ValueError("只支持单输入模型.")
    input_detail = tensor_metadata(inputs[0])
    if input_detail["shape"] != [1, 3, 640, 640]:
        raise ValueError(f"输入布局不是 NCHW: {input_detail}.")
    outputs = []
    for index, detail in enumerate(reader.get_output_tensor_details()):
        output = tensor_metadata(detail)
        if output["shape"] not in OUTPUT_SHAPES:
            raise ValueError(f"输出布局不匹配: {output}.")
        output["semantic"] = OUTPUT_NAMES[OUTPUT_SHAPES.index(output["shape"])]
        output["index"] = index
        outputs.append(output)
    if sorted(item["semantic"] for item in outputs) != sorted(OUTPUT_NAMES):
        raise ValueError("输出头缺失或重复.")
    metadata = {"input": input_detail, "outputs": outputs,
                "calibration": calibration,
                "converter": mtk_converter.__version__,
                "native_layout": "NCHW_INT8_WIDTH_ALIGN16",
                "native_layout_board_verified": False}
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    contract = [input_detail] + outputs
    lines = []
    for item in contract:
        shape = item["shape"]
        lines.append(",".join(map(str, [item.get("semantic", "input"),
                                      item.get("index", 0), *shape[1:],
                                      item["scale"], item["zero_point"]])))
    (args.output.parent / "runtime_config.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("[OK] TFLite 与量化元数据已保存,原生输出布局仍需板端验证.")


if __name__ == "__main__":
    main()
