"""复用公共命令工具上传完整 Python 混合部署包,不执行板端入口."""

import json
import os
from pathlib import Path
import shlex

from tools.deploy import context
from tools.process import run_command


def main():
    """上传固定子图,CPU 参数,一帧输入和板端 Python 依赖代码."""
    ctx = context(__file__)
    manifest = json.loads((ctx.work / "contract.json").read_text(encoding="utf-8"))
    config = ctx.work / "config.json"
    config.write_text(json.dumps(ctx.config, ensure_ascii=False, indent=2), encoding="utf-8")
    model_files = [ctx.work / name for name in ("contract.json", "sparse_weights.npz", "geometry.npz")]
    for name, item in manifest["stages"].items():
        model_files.append(ctx.work / f"{name}.onnx")
        if item["backend"] == "mtk_npu":
            model_files.append(ctx.work / item["file"])
    groups = {"": [ctx.model / "deploy/compile.sh", ctx.model / "deploy/evaluate.sh",
                     ctx.model / "deploy/requirements-evaluate.txt", config],
              "models": model_files, "board": list((ctx.model / "deploy/board").glob("*.py")),
              "board/tools": [ctx.repo / "tools" / name for name in
                              ("__init__.py", "runtime.py", "board.py", "evaluation.py")],
              "examples/input": [ctx.work / "sample.npz", ctx.work / "sample.json"]}
    for files in groups.values():
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
    host, destination = os.environ["MTK_BOARD_HOST"], ctx.config["board"]["deploy_dir"]
    options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    directories = [str(Path(destination) / folder) for folder in groups]
    run_command(["ssh", *options, host, "mkdir -p " + " ".join(map(shlex.quote, directories))])
    for folder, files in groups.items():
        run_command(["scp", *options, *files, f"{host}:{destination}/{folder}/"])
    print("[上传完成] 待板端检查.请登录开发板,在本模型部署目录执行 bash compile.sh.", flush=True)


if __name__ == "__main__":
    main()
