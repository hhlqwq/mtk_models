"""compile 后准备固定小样本数值参考,仅保存在主机本地并上传板端."""

from datetime import date
import io
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

import numpy as np

from tools.deploy import context
from tools.output_metrics import (
    ReferenceComplete, ReferenceWriter, finish_reference, record_reference,
)
from tools.process import run_command
from tools.files import sha256


def classification(ctx):
    """分类小样本直接复用板端预处理,不依赖全量标签或评价循环."""
    import onnxruntime as ort

    sys.path.insert(0, str(ctx.model / "deploy/board"))
    if ctx.config["name"] == "resnet101":
        from resnet101 import preprocess

        filename, count = "resnet101.onnx", 3
    else:
        from infer import floating_preprocess as preprocess

        filename, count = "vit_b_16_mtk_compatible.onnx", 5
    images = ctx.dataset / "val"
    if not images.is_dir():
        images = ctx.dataset / "ILSVRC2012_img_val"
    paths = [images / f"ILSVRC2012_val_{index:08d}.JPEG" for index in range(1, count + 1)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"分类小样本输入不存在: {path}.")
    source = ctx.output / filename
    board_inputs = None
    if ctx.config["name"] == "vit_base_patch16_224":
        # OpenCV 在不同 CPU 上的缩放结果可能不同,用板端实际前处理生成数值参考.
        script = "\n".join((
            "import hashlib, io, json, sys",
            "from pathlib import Path",
            "import numpy as np",
            f"root = Path({ctx.config['board']['deploy_dir']!r})",
            "sys.path.insert(0, str(root / 'board'))",
            "from infer import floating_preprocess",
            "config = json.loads((root / 'config.json').read_text())",
            "images = Path(config['board']['dataset_dir'])",
            "images = images / 'val' if (images / 'val').is_dir() else images / 'ILSVRC2012_img_val'",
            f"paths = [images / f'ILSVRC2012_val_{{index:08d}}.JPEG' for index in range(1, {count + 1})]",
            "values = {path.name: floating_preprocess(path).astype(np.float32) for path in paths}",
            "values['files'] = np.array(json.dumps({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}))",
            "buffer = io.BytesIO()",
            "np.savez(buffer, **values)",
            "sys.stdout.buffer.write(buffer.getvalue())",
        ))
        payload = subprocess.check_output([
            "ssh", "-o", "BatchMode=yes", os.environ["MTK_BOARD_HOST"],
            "PYTHONDONTWRITEBYTECODE=1 python3 -c " + shlex.quote(script)])
        board_inputs = np.load(io.BytesIO(payload), allow_pickle=False)
        identities = json.loads(str(board_inputs["files"]))
        if any(identities.get(path.name) != sha256(path) for path in paths):
            raise ValueError("ViT 数值参考的主机和板端原始图片不一致.")
    model = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    for path in paths:
        tensor = (board_inputs[path.name] if board_inputs is not None
                  else np.asarray(preprocess(path), dtype=np.float32))
        record_reference(source, [tensor], {"output": model.run(None, {model.get_inputs()[0].name: tensor})[0]})


def language(ctx, path):
    """只对照示例首个 Prefill 块的有效位置,不生成全量题库张量."""
    import torch
    from tools.llm.evaluate_mmlu import load_reference

    package = ctx.output / "models"
    model = load_reference(Path(ctx.config["model"]["source"]))
    writer = ReferenceWriter(path)
    languages = ("en",) if ctx.config["name"] == "minicpm5_2b" else ("zh", "en")
    for language in languages:
        prompt = package / "scripts/prompts" / f"{language}_demo.tokens.txt"
        ids = np.array([[int(value) for value in prompt.read_text().split()][:128]], dtype=np.int64)
        with torch.inference_mode():
            output = model(torch.from_numpy(ids), use_cache=False).logits.numpy()
        writer.add(Path(ctx.config["model"]["source"]) / "config.json", [ids], {"output": output})
    writer.finish()


def main():
    """复用现有浮点输入协议,达到五次输入即结束,不计算全量任务精度."""
    deploy = Path(sys.argv[1]).resolve()
    sys.argv = [str(deploy / "host/evaluate_onnx.py")]
    os.environ.update(MTK_EVAL_SCOPE="subset", MTK_PREPARE_OUTPUT_REFERENCE="1",
                      MTK_EVALUATION_WORK="1")
    ctx = context(deploy / "host/evaluate_onnx.py")
    local = Path("/tmp/hailongcodex") / date.today().isoformat() / ctx.config["name"] / "compile"
    local.mkdir(parents=True, exist_ok=True)
    reference = local / "output_reference.npz"
    os.environ["TMPDIR"] = str(local)
    os.environ["MTK_OUTPUT_REFERENCE"] = str(reference)
    os.environ["MTK_OUTPUT_MODEL_DIRS"] = json.dumps([
        str(ctx.output / "models" if ctx.config["name"] in
            ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b") else ctx.output)])
    sys.path.insert(0, str(deploy / "host"))
    name = ctx.config["name"]
    try:
        if name in ("resnet101", "vit_base_patch16_224"):
            classification(ctx)
        elif name in ("minicpm5_2b", "deepseek_r1_distill_qwen_1_5b"):
            language(ctx, reference)
        else:
            if name == "yolov8n":
                sys.argv += ["--onnx", str(ctx.output / "yolov8n.onnx")]
            runpy.run_path(str(deploy / "host/evaluate_onnx.py"), run_name="__main__")
    except ReferenceComplete:
        pass
    finish_reference()
    if not reference.is_file():
        raise FileNotFoundError("未生成 compile 小样本浮点参考.")
    host = os.environ["MTK_BOARD_HOST"]
    destination = ctx.config["board"]["deploy_dir"] + "/board/output_reference.npz"
    run_command(["scp", "-o", "BatchMode=yes", reference, f"{host}:{destination}"])
    print("[小样本参考完成] 已上传,板端 compile 比较并打印,全量 evaluate 不读取.", flush=True)


if __name__ == "__main__":
    main()
