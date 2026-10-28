"""独立全量 FP32 参考,已有同协议参考则直接复用,不重新编译."""

from datetime import date
import json
from pathlib import Path
import sys

from tools.deploy import context
from tools.evaluation import is_test

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from evaluate import evaluate_dataset, identity
from scoring import score_predictions


def main():
    """使用已有四个 ONNX 和相同 CPU 稀疏主干计算完整浮点参考."""
    ctx = context(__file__)
    samples = json.loads((ctx.work / "samples.json").read_text(encoding="utf-8"))["frames"]
    if not is_test() and len(samples) != 6019:
        raise ValueError("完整浮点参考必须覆盖 val 6019 帧.")
    package = ctx.output / "models"
    fingerprint = identity(package)
    tokens = [sample["token"] for sample in samples]
    target = ctx.work / "onnx_reference.json"
    if target.is_file():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous.get("identity") == fingerprint and previous.get("sample_tokens") == tokens:
            print("[参考复用] 已有同模型同协议完整 FP32 参考,无需重复运行.", flush=True)
            return
    temporary = Path("/tmp/hailongcodex") / date.today().isoformat() / "mit_bevfusion_reference"
    predictions, _, _, _ = evaluate_dataset(package, ctx.dataset, samples,
                                           temporary / "predictions.json", floating=True)
    accuracy = score_predictions(ctx.work / "ground_truth.json", predictions)
    target.write_text(json.dumps({"identity": fingerprint, "sample_tokens": tokens, "accuracy": accuracy}),
                      encoding="utf-8")
    print("[参考完成] 将 onnx_reference.json 单独放入板端 models/,不需要重新上传或编译模型.", flush=True)


if __name__ == "__main__":
    main()
