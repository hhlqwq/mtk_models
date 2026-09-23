"""将 DLA、量化参数、ONNX 和基线绑定到同一构建清单."""

import argparse
import json
from pathlib import Path

from fastsam_utils import sha256_file


def main():
    """核对转换来源链后生成板端部署清单."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.model_dir
    quantization = json.loads((root / "model_int8.json").read_text(encoding="utf-8"))
    exported = json.loads((root / "export_manifest.json").read_text(encoding="utf-8"))
    onnx_digest = sha256_file(root / "model_fp32.onnx")
    if onnx_digest != quantization["onnx_sha256"] or onnx_digest != exported["onnx_sha256"]:
        raise ValueError("导出与量化 ONNX 来源不一致.")
    if sha256_file(root / "model_int8.tflite") != quantization["tflite_sha256"]:
        raise ValueError("编译输入 TFLite 与量化元数据不一致.")
    names = ["model_int8.dla", "model_int8.json", "runtime_config.csv", "model_fp32.onnx",
             "pytorch_reference.npz", "export_manifest.json"]
    manifest = {"arch": "mdla5.3", "suppress_output": True, "disallow_bridge": True,
                "weights_sha256": exported["weights_sha256"],
                "files": {name: sha256_file(root / name) for name in names}}
    (root / "deployment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[MANIFEST] 部署模型与基线已绑定.")


if __name__ == "__main__":
    main()
