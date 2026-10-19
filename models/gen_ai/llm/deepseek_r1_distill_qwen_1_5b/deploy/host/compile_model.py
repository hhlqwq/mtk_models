"""独立编译 DeepSeek FP16 分片并记录部署产物清单."""

import json

from tools.deploy import arguments, compile_dla
from tools.files import sha256


def main():
    """逐个编译 build 中间模型,最终 DLA 保存到模型资源目录."""
    ctx = arguments(__file__)
    manifest = json.loads((ctx.output / "export_manifest.json").read_text())
    for graph in manifest["graphs"]:
        name = graph["name"]
        compile_dla(ctx, ctx.work / f"{name}.tflite", ctx.output / f"{name}.dla",
                    ("--suppress-input",))
    paths = [ctx.output / f"{graph['name']}.dla" for graph in manifest["graphs"]]
    paths += [ctx.output / "export_manifest.json"]
    if not manifest["probe_only"]:
        paths += [ctx.output / "embedding_fp16.bin"]
    report = {"architecture": "mdla5.3", "precision": "FP16", "disallow_bridge": True,
              "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                        for path in paths]}
    (ctx.work / "artifact_manifest.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
