"""上传 DeepSeek 模型、桥接库和配置到原板端目录."""

from tools.deploy import arguments, upload
from tools.evaluation import is_test


def main():
    """只更换快速样例文件,板端部署路径不追加 _test."""
    ctx = arguments(__file__)
    suffix = "_test" if is_test() else ""
    files = [*ctx.output.glob("*.dla"), ctx.output / "embedding_fp16.bin",
             ctx.output / "export_manifest.json", ctx.work / "artifact_manifest.json",
             ctx.output / "source_manifest.json", ctx.output / "tokenizer.json"]
    sample = ctx.output / f"samples{suffix}.json"
    snapshot = ctx.work / "samples.json"
    snapshot.write_bytes(sample.read_bytes())
    files.append(snapshot)
    upload(ctx, files, [ctx.work / "libneuron_bridge.so"])


if __name__ == "__main__":
    main()
