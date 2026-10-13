"""执行全 W4A16 量化,传播 SDK 异常并核对实际权重位宽."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path




@contextmanager
def strict_temp_file(filename):
    """保留 SDK 临时文件清理行为,同时向调用者传播转换异常."""
    try:
        yield
    finally:
        Path(filename).unlink(missing_ok=True)


def verify_weight_precision(args, utils):
    """逐层核对全部投影权重,拒绝非 W4A16 配置."""
    if args.precision != ["asym4W_sym16A"] or args.tail_precision not in (
            None, "asym4W_sym16A"):
        raise ValueError("本项目仅允许全 W4A16 量化.")
    if args.lora_config or args.tail_config or args.int8_cache or args.format != "tflite":
        raise ValueError("权重核对仅支持标准 TFLite 模型和 INT16 Cache.")
    config = json.loads(Path(args.config).read_text())
    mlp_precision = args.precision[-1]
    precision_name = args.precision[0]
    prefix = (f"{utils.get_exp_name(args.config)}_{precision_name}"
              f"_{args.calibration_method}")
    if args.weight_optimization:
        prefix += f"_{args.weight_optimization}"
    if args.extra_converter_options:
        prefix += f"_{Path(args.extra_converter_options).stem}"
    if config.get("rotate"):
        prefix += f"_rotate_{config['rotate_mode']}_{config['rotate_seed']}"
    layers = config["num_hidden_layers"]
    counts = {"attention_weights": 0, "mlp_weights": 0, "tail_weights": 0}
    for index in range(layers + 1):
        path = Path("quant_configs/ptq") / prefix / f"{prefix}_{index}.json"
        specs = json.loads(path.read_text())["precision_specs"]
        seen = set()
        for group in specs:
            for name in group["wgt_names"]:
                if name in seen:
                    raise ValueError(f"同一权重出现于多个精度组: {index},{name}")
                seen.add(name)
                if index == layers:
                    expected = args.tail_precision or mlp_precision
                    kind = "tail_weights"
                elif "/self_attn/" in name:
                    expected, kind = args.precision[0], "attention_weights"
                elif "/mlp/" in name:
                    expected, kind = mlp_precision, "mlp_weights"
                else:
                    raise ValueError(f"出现未支持的投影权重: {index},{name}")
                if group["precision_name"] != expected:
                    raise ValueError(f"权重精度与请求不一致: {index},{name}")
                counts[kind] += 1
    if counts != {"attention_weights": layers * 4,
                  "mlp_weights": layers * 3, "tail_weights": 1}:
        raise ValueError(f"标准 Attention/MLP 投影权重数量不符: {counts}")
    Path("weight_precision_audit.json").write_text(json.dumps({
        "parts": layers + 1, "weight_groups_verified": counts,
        "attention": args.precision[0], "mlp": mlp_precision,
        "lm_head": args.tail_precision or mlp_precision,
    }, indent=2) + "\n")
    print(f"[实际权重精度核对完成] {counts}", flush=True)


def main():
    """执行固定 SDK 的进程内兼容逻辑,不修改安装文件或官方模型."""
    from mtk_llm_sdk import ptq
    from mtk_llm_sdk.utils import utils

    args = ptq.get_argument_parser().parse_args()
    source_path = Path(ptq.__file__)
    source = source_path.read_text()
    if args.precision != ["asym4W_sym16A"] or args.tail_precision not in (
            None, "asym4W_sym16A"):
        raise ValueError("本项目仅允许全 W4A16,请排查误差来源而非提高权重位宽.")
    utils.temp_file = strict_temp_file
    Path("quantization_adapter.json").write_text(json.dumps({
        "sdk_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "adapter": "w4_only_strict_exceptions_weight_audit",
        "installed_sdk_modified": False,
    }, indent=2) + "\n")
    print("[SDK 兼容入口] 使用独立进程,转换异常将使任务失败.", flush=True)
    ptq.main()
    verify_weight_precision(args, utils)


if __name__ == "__main__":
    main()
