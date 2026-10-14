"""离线导出 FastSAM-s 原始分割头,保留官方前向结果用于一致性检查."""

import argparse
import sys
import copy
import json
import shutil
import subprocess
import types
from pathlib import Path


def raw_forward(self, features):
    """分离三个尺度的框、分类、系数和原型,避免将 DFL/NMS 编入 NPU."""
    outputs = []
    for index in range(3):
        outputs.extend((self.cv2[index](features[index]),
                        self.cv3[index](features[index]),
                        self.cv4[index](features[index])))
    outputs.append(self.proto(features[0]))
    return tuple(outputs)


def load_model(weights):
    """只加载已在本地放置的可信官方权重,禁止自动下载与依赖安装."""
    import torch
    import ultralytics
    if ultralytics.__version__ != "8.0.111":
        raise RuntimeError(f"要求 ultralytics==8.0.111,实际 {ultralytics.__version__}.")
    if not weights.is_file():
        raise FileNotFoundError(f"请离线放置官方权重: {weights}")
    checkpoint = torch.load(str(weights), map_location="cpu")
    model = checkpoint.get("ema") or checkpoint["model"]
    model = model.float().cpu().eval()
    # 兼容早期 PyTorch 序列化的 Upsample; None 保持旧版默认缩放语义.
    for module in model.modules():
        if isinstance(module, torch.nn.Upsample) and not hasattr(
                module, "recompute_scale_factor"):
            module.recompute_scale_factor = None
    head = model.model[-1]
    if (head.nc, head.nm, head.reg_max, head.nl) != (1, 32, 16, 3):
        raise ValueError("权重不是预期的 FastSAM 分割结构.")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if not 10_000_000 < parameters < 15_000_000:
        raise ValueError(f"要求 FastSAM-s,参数量不匹配: {parameters}.")
    if list(head.stride.tolist()) != [8, 16, 32]:
        raise ValueError("检测头 stride 不匹配.")
    head.export = False
    return model


def main():
    """检查前向等价并导出固定 opset 13 模型."""
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--config", type=Path, help="使用带备注的 YAML 配置.")
    parser.add_argument("--work-dir", type=Path, help="复用指定的中间文件目录.")
    parser.add_argument("--inside-container", action="store_true", help=argparse.SUPPRESS)
    options, _ = parser.parse_known_args()
    parser.add_argument("-h", "--help", action="help", help="显示参数说明.")
    paths = None
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
        paths = {"weights": model / config["model"]["weights"],
                 "image": dataset / config["data"]["sample_image"], "work": work,
                 "output": model / config["model"]["output_dir"]}
    parser.add_argument("--weights", type=Path, required=paths is None,
                        default=paths["weights"] if paths else None)
    parser.add_argument("--image", type=Path, required=paths is None,
                        default=paths["image"] if paths else None)
    parser.add_argument("--output-dir", type=Path, required=paths is None,
                        default=paths["work"] / "export" if paths else None)
    args = parser.parse_args()
    if paths and not args.inside_container and not Path("/.dockerenv").exists():
        container = config["build"]["container"]
        subprocess.run(["docker", "exec", "-e", "MTK_MODEL_STORAGE_ROOT", "-e", "MTK_DATASET_ROOT", container, "mkdir", "-p", str(work / "tmp"), str(work / "cache")], check=True)
        print(f"[导出] 在 Docker {container} 中执行.", flush=True)
        subprocess.run(["docker", "exec", "-e", "MTK_MODEL_STORAGE_ROOT", "-e", "MTK_DATASET_ROOT", "-e", "PYTHONDONTWRITEBYTECODE=1",
                        "-e", f"TMPDIR={work / 'tmp'}", "-e", f"XDG_CACHE_HOME={work / 'cache'}",
                        "-e", f"TORCH_HOME={work / 'cache/torch'}", "-w", str(work), container,
                        "python", str(Path(__file__).resolve()), "--inside-container",
                        "--config", str(args.config.resolve()), "--work-dir", str(work),
                        "--weights", str(args.weights.resolve()), "--image", str(args.image.resolve()),
                        "--output-dir", str(args.output_dir.resolve())], check=True)
        return
    # SDK 与模型依赖只在实际导出环境加载,89 主机只负责启动容器命令.
    import cv2
    import numpy as np
    import onnx
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
    from fastsam_utils import OUTPUT_NAMES, OUTPUT_SHAPES, decode_heads, preprocess
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[1/4] 离线加载官方 FastSAM-s 权重.", flush=True)
    original = load_model(args.weights)
    raw_model = copy.deepcopy(original)
    raw_model.model[-1].forward = types.MethodType(raw_forward, raw_model.model[-1])
    tensor, geometry = preprocess(cv2.imread(str(args.image)))
    with torch.no_grad():
        reference = original(torch.from_numpy(tensor))
        outputs = raw_model(torch.from_numpy(tensor))
    arrays = [value.numpy() for value in outputs]
    boxes, scores, coefficients = decode_heads(arrays)
    xywh = np.concatenate(((boxes[:, :2] + boxes[:, 2:]) / 2,
                           boxes[:, 2:] - boxes[:, :2]), axis=1)
    decoded = np.concatenate((xywh, scores[:, None], coefficients), axis=1).T[None]
    expected = reference[0].numpy()
    np.testing.assert_allclose(decoded, expected, atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(arrays[-1], reference[1][-1].numpy(), atol=1e-6)
    print("[2/4] 原始头 CPU 解码与官方前向一致,导出 ONNX.", flush=True)
    onnx_path = args.output_dir / "model_fp32.onnx"
    torch.onnx.export(raw_model, torch.from_numpy(tensor), str(onnx_path),
                      input_names=["images"], output_names=OUTPUT_NAMES,
                      opset_version=13, do_constant_folding=True)
    exported = onnx.load(str(onnx_path))
    onnx.checker.check_model(exported)
    shapes = [[dim.dim_value for dim in item.type.tensor_type.shape.dim]
              for item in exported.graph.output]
    if shapes != OUTPUT_SHAPES:
        raise ValueError(f"ONNX 输出形状错误: {shapes}.")
    print("[3/4] 保存 PyTorch 输入与输出基线.", flush=True)
    np.savez(args.output_dir / "pytorch_reference.npz", images=tensor,
             **dict(zip(OUTPUT_NAMES, arrays)))
    manifest = {
        "weights": str(args.weights),
        "implementation": "ultralytics/ultralytics@v8.0.111",
        "torch": torch.__version__, "onnx": onnx.__version__,
        "image": str(args.image), "geometry": geometry,
        "output_names": OUTPUT_NAMES, "output_shapes": OUTPUT_SHAPES,
        "raw_decode_max_absolute_error": float(np.max(np.abs(decoded - expected))),
        "onnx_runtime_verified": False,
    }
    (args.output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if paths:
        paths["output"].mkdir(parents=True, exist_ok=True)
        destination = paths["output"] / "model_fp32.onnx"
        if onnx_path.resolve() != destination.resolve():
            shutil.copy2(onnx_path, destination)
    print("[4/4] 导出完成,ONNX 数值验证和 NPU 验证仍需单独执行.", flush=True)


if __name__ == "__main__":
    main()
