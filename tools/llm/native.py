"""LLM 原生独立步骤的公共路径和执行环境,不分派转换阶段."""

import argparse
import os
from pathlib import Path

from tools.deploy import arguments
from tools.process import run_command


def settings(ctx, *, python_executable=None):
    """读取模型配置,可显式使用调用脚本的 Python 解释器."""
    config = ctx.config["native"]
    name = "MiniCPM5-2B" if ctx.config["name"] == "minicpm5_2b" else "DeepSeek-R1-Distill-Qwen-1.5B"
    work = Path(os.environ.get("NATIVE_WORK", ctx.output / config["work_dir"]))
    context = int(os.environ.get("CONTEXT_SIZE", ctx.config["conversion"]["context"]))
    prefill = int(os.environ.get("PREFILL_TOKENS", config["prefill"]))
    batches = int(os.environ.get("CALIBRATION_BATCHES", config["calibration_batches"]))
    rotate = int(os.environ.get("ROTATE_WEIGHTS", int(config["rotate"])))
    precisions = [os.environ.get(key, config[field]) for key, field in (
        ("ATTENTION_PRECISION", "attention_precision"), ("MLP_PRECISION", "mlp_precision"),
        ("TAIL_PRECISION", "tail_precision"))]
    if any(value != "asym4W_sym16A" for value in precisions):
        raise ValueError("仅支持全 W4A16,禁止提高权重位宽替代误差排查.")
    if context < prefill or prefill < 1 or batches < 1 or rotate not in (0, 1):
        raise ValueError("上下文、Prefill、校准批次或旋转参数无效.")
    optimization = os.environ.get("WEIGHT_OPTIMIZATION", config["weight_optimization"])
    extra = os.environ.get("WEIGHT_OPT_CONFIG", config["weight_opt_config"])
    if optimization not in ("none", "hessian", "gradient") or (optimization == "none" and extra):
        raise ValueError("重优化设置不符合 none/hessian/gradient 协议.")
    prefix = f"{name}_asym4W_sym16A_Overall"
    if optimization != "none":
        prefix += "_" + optimization
    if extra:
        prefix += "_" + Path(extra).stem
    if rotate:
        prefix += "_rotate_hadamard_0"
    work.mkdir(parents=True, exist_ok=True)
    return argparse.Namespace(ctx=ctx, work=work, model=work / name,
        source=Path(os.environ.get("MODEL_SOURCE", ctx.config["model"]["source"])),
        python=Path(python_executable or os.environ["MTK_GAI_PYTHON"]), context=context, prefill=prefill,
        batches=batches, rotate=rotate,
        mask=float(os.environ.get("MASK_VALUE", config["mask_value"])),
        optimization=optimization, extra=extra, prefix=prefix,
        quantized=work / "tflite" / prefix,
        train=os.environ.get("CALIBRATION_TRAIN", config["calibration_train"]))


def python(native, *arguments):
    """使用进入容器时加载的专用 Python 环境,保留离线执行规则."""
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    run_command([native.python, *arguments], cwd=native.work, env=environment)
