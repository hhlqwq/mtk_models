"""在独立进程中兼容 SDK 2.7.5 的旋转混合精度分组,并传播转换异常."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


def normalize_rotation_groups(config):
    """合并旋转新增的 Attention 组,保持 SDK 原有 Attention/MLP 分配逻辑."""
    groups = config["precision_specs"]
    if len(groups) == 2:
        return config
    if len(groups) != 3 or any(group["precision_name"] != "FP" for group in groups):
        raise ValueError("旋转混合精度只兼容已确认的三个 FP 模板组.")
    mlp_indices = [index for index, group in enumerate(groups)
                   if group["wgt_names"] and all("/mlp/" in name
                                                for name in group["wgt_names"])]
    if len(mlp_indices) != 1:
        raise ValueError("无法唯一确定独立 MLP 精度组.")
    mlp_index = mlp_indices[0]
    other_indices = [index for index in range(3) if index != mlp_index]
    weights = [name for index in other_indices for name in groups[index]["wgt_names"]]
    if not any("/self_attn/" in name for name in weights) or not any(
            "/mlp/" in name for name in weights):
        raise ValueError("旋转新增组与已确认的 Attention/MLP 模板不符.")
    merged = {"precision_name": "FP", "estimated_macs": None}
    for key in ("wgt_names", "param_names", "act_names"):
        merged[key] = list(dict.fromkeys(name for index in other_indices
                                        for name in groups[index][key]))
    if not any(name in merged["act_names"] for name in ("inputs_embeds.1", "argument_1.1")):
        raise ValueError("Attention 模板缺少 SDK 用于识别的输入名称.")
    config["precision_specs"] = [groups[mlp_index], merged]
    # 同步重映射模板策略索引,保持原先优先顺序并去掉合并造成的重复.
    mapping = {index: int(index != mlp_index) for index in range(3)}
    for policy, indices in config.get("predefined_policies", {}).items():
        config["predefined_policies"][policy] = list(dict.fromkeys(mapping[index]
                                                                for index in indices))
    print("[SDK 兼容] 旋转三个模板组归并为 MLP/Attention,随后按官方逻辑拆分权重.",
          flush=True)
    return config


@contextmanager
def strict_temp_file(filename):
    """保留 SDK 临时文件清理行为,同时向调用者传播转换异常."""
    try:
        yield
    finally:
        Path(filename).unlink(missing_ok=True)


def main():
    """执行固定 SDK 的进程内兼容逻辑,不修改安装文件或官方模型."""
    from mtk_llm_sdk import ptq
    from mtk_llm_sdk.utils import utils

    source_path = Path(ptq.__file__)
    source = source_path.read_text()
    needle = ('                    assert len(quant_config[\'precision_specs\']) == 2, '
              'f"{quant_config_file}: {len(quant_config[\'precision_specs\'])}"')
    if source.count(needle) != 1:
        raise ValueError("SDK 源码与已确认的 2.7.5 分组逻辑不符,拒绝自动修改.")
    replacement = ("                    if not tail:\n"
                   "                        quant_config = normalize_rotation_groups(quant_config)")
    ptq.__dict__["normalize_rotation_groups"] = normalize_rotation_groups
    # 仅重新编译本进程模块; Tail 的任意组数均使用官方已有遍历逻辑.
    exec(compile(source.replace(needle, replacement), str(source_path), "exec"), ptq.__dict__)
    utils.temp_file = strict_temp_file
    Path("quantization_adapter.json").write_text(json.dumps({
        "sdk_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "adapter": "rotation_mixed_groups_v1_strict_exceptions",
        "installed_sdk_modified": False,
    }, indent=2) + "\n")
    print("[SDK 兼容入口] 使用独立进程,转换异常将使任务失败.", flush=True)
    ptq.main()


if __name__ == "__main__":
    main()
