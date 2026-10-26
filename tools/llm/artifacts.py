"""统一 LLM 编译产物命名与部署清单中的阶段路径."""

from pathlib import PurePosixPath


def dla_filename(model, layers, tokens, context):
    """按模型、精度、层数和实际输入形状生成可读文件名."""
    if min(layers, tokens, context) < 1 or tokens > context:
        raise ValueError("DLA 层数、输入长度或上下文无效.")
    if not model or any(value in model for value in ("/", "\\", "..")):
        raise ValueError("模型名称不能包含路径.")
    phase = "decode" if tokens == 1 else "prefill"
    return (f"{model}_asym4W_sym16A_{layers}layer_"
            f"{phase}_{tokens}t_{context}c.dla")


def dla_relative_path(manifest, phase):
    """读取阶段文件路径,未迁移部署包仍按原清单加载."""
    if phase not in ("prompt", "decode"):
        raise ValueError("未知 DLA 阶段.")
    name = manifest.get("model_files", {}).get(
        phase, f"{manifest['context']}c/{phase}.dla")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("DLA 路径必须位于部署包内.")
    return name
