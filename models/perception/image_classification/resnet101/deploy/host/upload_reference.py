"""上传已有完整 ONNX 参考,不重新评价或修改板端模型."""

import json
import math
import os
from pathlib import Path
import shlex
import sys

from tools.deploy import arguments, reference_path
from tools.files import read_json, require_file
from tools.process import run_command

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from resnet101 import MODEL, PROTOCOL, WEIGHTS, label_signature, read_labels, validate_contract


def main() -> None:
    """核对主机和现有板端模型版本,仅复制一份参考文件."""
    ctx = arguments(__file__)
    source = require_file(reference_path(ctx))
    reference = read_json(source)
    metadata = read_json(ctx.output / "build/quantization.json")
    validate_contract(metadata)
    required = {
        "model": MODEL, "weights": WEIGHTS, "status": "complete",
        "evaluation_scope": "full", "samples": 50000, "protocol": PROTOCOL, "metric": "Top-1",
        "onnx_sha256": metadata["onnx_sha256"], "pillow": metadata["pillow"],
        "labels_sha256": label_signature(read_labels(ctx.dataset / "val_labels_0based.txt")),
    }
    if any(reference.get(key) != value for key, value in required.items()):
        raise ValueError("已有浮点参考与模型、标签或评价协议不一致,拒绝上传.")
    if not math.isfinite(reference["accuracy"]) or not 0 <= reference["accuracy"] <= 1:
        raise ValueError("已有浮点参考 Top-1 无效.")
    host = os.environ["MTK_BOARD_HOST"]
    destination = Path(ctx.config["board"]["deploy_dir"]) / "models"
    options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    result = run_command(
        ["ssh", *options, host, "cat -- " + shlex.quote(str(destination / "quantization.json"))],
        capture_output=True, text=True,
    )
    deployed = json.loads(result.stdout)
    for key in ("model", "weights", "protocol", "onnx_sha256", "dla_sha256"):
        if deployed.get(key) != metadata[key]:
            raise ValueError(f"板端模型与本次参考版本不匹配: {key}.")
    run_command(["scp", *options, source, f"{host}:{destination}/onnx_reference.json"])



if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    main()
