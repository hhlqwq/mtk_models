"""compile 后准备固定小样本数值参考,仅保存在主机本地并上传板端."""

from datetime import date
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np

from tools.deploy import context
from tools.output_metrics import (
    ReferenceComplete, ReferenceWriter, finish_reference, record_reference,
)
from tools.process import run_command


def classification(ctx):
    """ResNet 复用板端三张展示输入和相同的预处理函数."""
    import onnxruntime as ort

    sys.path.insert(0, str(ctx.model / "deploy/board"))
    from resnet101 import preprocess, validation_images

    source = ctx.output / "resnet101.onnx"
    model = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    for path in validation_images(ctx.dataset)[:3]:
        tensor = preprocess(path)
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
        if name == "resnet101":
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
