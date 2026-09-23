"""部署前校验 DLA 与样例图属于当前导出清单."""

import argparse
import json
from pathlib import Path

from fastsam_utils import sha256_file


def main():
    """阻止模型、张量量化参数和原始图片发生交叉混用."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    deployment = json.loads((args.model_dir / "deployment_manifest.json").read_text(
        encoding="utf-8"))
    exported = json.loads((args.model_dir / "export_manifest.json").read_text(
        encoding="utf-8"))
    for name, expected in deployment["files"].items():
        if Path(name).name != name or sha256_file(args.model_dir / name) != expected:
            raise ValueError(f"部署文件哈希不匹配: {name}")
    if sha256_file(args.image) != exported["image_sha256"]:
        raise ValueError("冒烟图片与 PyTorch 导出基线不一致.")
    print("[VERIFY] DLA、量化配置、原始图片与导出记录一致.")


if __name__ == "__main__":
    main()
