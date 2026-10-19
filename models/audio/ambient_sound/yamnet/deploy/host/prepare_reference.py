"""整理音频浮点参考与少量示例,区分快速验证和全量评测."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from export_evaluate import load_records, mapping
from tools.evaluation import add_test_argument, audio_records, is_test, scope_metadata


def main():
    """读取本次两后端评测报告,校验范围并保存同范围参考."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    add_test_argument(parser)
    args = parser.parse_args()
    records = audio_records(load_records(args.dataset))
    labels = args.models_dir / "upstream/yamnet_class_map.csv"
    reference = {}
    for backend in ("tensorflow", "onnx"):
        report = json.loads((args.work / backend / "summary.json").read_text())
        if (report.get("samples") != len(records) or
                report.get("evaluation_scope", "full") != scope_metadata()["evaluation_scope"]):
            raise ValueError(f"{backend} 参考报告与本次样本范围不一致.")
        if is_test() and report.get("sample_ids_sha256") != scope_metadata()["sample_ids_sha256"]:
            raise ValueError(f"{backend} 参考报告的样本指纹不一致.")
        reference[backend] = report
    reference["annotations_sha256"] = hashlib.sha256(
        (args.dataset / "meta/esc50.csv").read_bytes()).hexdigest()
    reference["mapping"] = mapping(labels)
    reference["audio_sha256"] = {
        row["filename"]: hashlib.sha256(
            (args.dataset / "audio" / row["filename"]).read_bytes()).hexdigest()
        for row in records}
    reference.update(scope_metadata())
    filename = "fp32_reference_test.json" if is_test() else "fp32_reference.json"
    (args.models_dir / filename).write_text(json.dumps(reference, indent=2))
    shutil.copy2(labels, args.models_dir / "yamnet_class_map.csv")
    inputs = args.model_root / "examples/input"
    inputs.mkdir(parents=True, exist_ok=True)
    samples = []
    for number, target in enumerate((0, 39, 42), 1):
        row = next(row for row in records if int(row["target"]) == target)
        shutil.copy2(args.dataset / "audio" / row["filename"], inputs / f"sample_{number}.wav")
        samples.append(row)
    (inputs / "samples.json").write_text(json.dumps(samples, indent=2))
    print(f"[参考] {args.models_dir / filename}", flush=True)


if __name__ == "__main__":
    main()
