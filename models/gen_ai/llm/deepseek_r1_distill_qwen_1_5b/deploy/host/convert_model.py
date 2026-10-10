"""编译自行导出的 Qwen2 分片,不允许 CPU 桥接或静默回退."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import mtk_converter


def artifact_record(path):
    """记录部署文件的真实大小与 SHA-256,支持分块读取大文件."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"name": path.name, "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def convert_model(args):
    """逐个转换与编译静态分片,保留每个阶段的日志和返回码."""
    manifest = json.loads(
        (args.models / "export_manifest.json").read_text(encoding="utf-8"))
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = (
        str(args.ncc_root / "lib") + ":" +
        environment.get("LD_LIBRARY_PATH", ""))
    logs = args.models / "compile_logs"
    logs.mkdir(exist_ok=True)
    for index, graph in enumerate(manifest["graphs"], 1):
        name = graph["name"]
        print(f"[转换 {index}/{len(manifest['graphs'])}] {name}", flush=True)
        tflite = args.models / f"{name}.tflite"
        converter = mtk_converter.OnnxConverter.from_model_proto_file(
            str(args.models / f"{name}.onnx"))
        converter.quantize = False
        converter.convert_to_tflite(str(tflite))
        command = [str(args.ncc_root / "bin/ncc-tflite"),
                   "--arch=mdla5.3", "--suppress-input", "--suppress-output",
                   "--disallow-bridge", str(tflite), "--dla-file",
                   str(args.models / f"{name}.dla")]
        print(f"[编译] {name},MDLA 5.3,禁止桥接.", flush=True)
        with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        if not (args.models / f"{name}.dla").stat().st_size:
            raise RuntimeError(f"编译产物为空: {name}")
    names = [f"{graph['name']}.dla" for graph in manifest["graphs"]]
    names.append("export_manifest.json")
    if not manifest["probe_only"]:
        names.append("embedding_fp16.bin")
    print("[归档] 校验部署产物大小与 SHA-256.", flush=True)
    artifacts = {"architecture": "mdla5.3", "precision": "FP16",
                 "disallow_bridge": True,
                 "files": [artifact_record(args.models / name) for name in names]}
    (args.models / "artifact_manifest.json").write_text(
        json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[编译] 全部分片完成,实际 I/O 类型与布局需板端验证.", flush=True)


def parse_args():
    """解析静态模型目录与 NCC 根目录."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--ncc-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    convert_model(parse_args())
