"""执行全 W4A16 量化,传播 SDK 异常并核对实际权重位宽."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

from mtk_converter.python.converters.tflite import importer
from mtk_converter.python.tools import builder
from mtk_converter.python.utils import tensor_utils
from mtk_llm_sdk import ptq
from mtk_llm_sdk import shape_fixer
from mtk_llm_sdk.utils import utils




@contextmanager
def strict_temp_file(filename):
    """传播 SDK 异常,清理临时文件并保留可审查的临时目录."""
    try:
        yield
    finally:
        path = Path(filename)
        if not path.is_dir():
            path.unlink(missing_ok=True)


def verify_tflite_precision(args, config, prefix):
    """直接读取常规量化图的权重编码和接口类型,不执行推理."""

    layers = config["num_hidden_layers"]
    counts = {"attention_weights": 0, "mlp_weights": 0, "tail_weights": 0}
    parts = []
    for index in range(layers + 1):
        path = Path("tflite") / prefix / f"{prefix}_{index}.tflite"
        raw = path.read_bytes()
        model = importer.import_from_model_buffer(raw)
        graph = model.subgraphs[0]
        tensors = graph.tensor_map
        for name in list(graph.inputs) + list(graph.outputs):
            tensor = tensors[name]
            dtype = tensor.DESCRIPTOR.fields_by_name["type"].enum_type.values_by_number[tensor.type].name
            if dtype != "DT_INT16":
                raise ValueError(f"量化接口不是 INT16: {index},{name},{dtype}")
        verified = []
        for name, tensor in tensors.items():
            if not name.endswith("/weight"):
                continue
            if "/self_attn/" in name:
                kind = "attention_weights"
            elif "/mlp/" in name:
                kind = "mlp_weights"
            elif "lm_head/" in name:
                kind = "tail_weights"
            else:
                continue
            dtype = tensor.DESCRIPTOR.fields_by_name["type"].enum_type.values_by_number[tensor.type].name
            if dtype != "DT_INT4":
                raise ValueError(f"投影权重不是 INT4: {index},{name},{dtype}")
            counts[kind] += 1
            verified.append(name)
        parts.append({"index": index, "sha256": hashlib.sha256(raw).hexdigest(),
                      "verified_weights": verified})
        print(f"[实际 TFLite 位宽核对] {index + 1}/{layers + 1}", flush=True)
        del model, graph, tensors, raw
    expected = {"attention_weights": layers * 4,
                "mlp_weights": layers * 3, "tail_weights": 1}
    if counts != expected:
        raise ValueError(f"实际投影权重数量不符: {counts}")
    Path("weight_precision_audit.json").write_text(json.dumps({
        "parts": layers + 1, "weight_groups_verified": counts,
        "attention": args.precision[0], "mlp": args.precision[0],
        "lm_head": args.tail_precision or args.precision[0],
        "evidence": "actual_tflite_int4_weights_int16_interfaces",
        "graphs": parts,
    }, indent=2) + "\n")


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
    if not (Path("quant_configs/ptq") / prefix).is_dir():
        verify_tflite_precision(args, config, prefix)
        return
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


def fix_native_shapes(folder=None, shapes=None):
    """导出两张静态图,显式转换不同 A16 尺度并传播 SDK 异常."""

    if folder is None:
        parser = argparse.ArgumentParser(description="导出 Prefill/Decode 单分片静态图.")
        parser.add_argument("folder", type=Path)
        parser.add_argument("shapes", nargs=2)
        parser.add_argument("-n", type=int, default=1)
        args = parser.parse_args(sys.argv[2:])
    else:
        args = argparse.Namespace(folder=Path(folder), shapes=list(shapes), n=1)
    if len(args.shapes) != 2:
        raise ValueError("必须同时指定 Prefill 和 Decode 两个形状.")
    if args.n != 1:
        raise ValueError("本项目每阶段只允许一个分片.")
    original = builder.ModelBuilder.import_subgraph
    adapters = []

    def import_with_requantize(self, filename_or_object, input_name_mappings=None,
                              subgraph_idx=0):
        """仅为同形状 INT16 线性量化接口插入真实重量化算子."""
        if isinstance(filename_or_object, (str, Path)):
            model = builder.nrp_pb2.ModelProto()
            model.ParseFromString(Path(filename_or_object).read_bytes())
        else:
            model = filename_or_object
        mappings = dict(input_name_mappings or {})
        graph = model.subgraphs[subgraph_idx]
        for source_name, target_name in list(mappings.items()):
            source = graph.tensor_map[source_name]
            target = self._cur_subgraph.tensor_map[target_name]
            if tensor_utils.have_same_type_and_quant_param([source, target]):
                continue
            if tensor_utils.get_shape(source) != tensor_utils.get_shape(target):
                raise ValueError(f"分片接口形状不一致: {source_name},{target_name}")
            for tensor in (source, target):
                dtype = tensor.DESCRIPTOR.fields_by_name["type"].enum_type.values_by_number[tensor.type].name
                if dtype != "DT_INT16" or len(tensor.quant_param.linear.scale_vals) != 1:
                    raise ValueError("仅支持 INT16 单尺度接口转换.")
            name = f"{target_name}_requant_{len(adapters)}"
            self._cur_subgraph.tensor_map[name].CopyFrom(source)
            self.op("Requantize", [target_name], [name])
            mappings[source_name] = name
            row = {"source": target_name, "target": source_name,
                   "input_scale": target.quant_param.linear.scale_vals[0],
                   "output_scale": source.quant_param.linear.scale_vals[0]}
            adapters.append(row)
            print(f"[分片 A16 尺度转换] {row}", flush=True)
        return original(self, model, mappings, subgraph_idx)

    builder.ModelBuilder.import_subgraph = import_with_requantize
    original_temp_file = utils.temp_file
    utils.temp_file = strict_temp_file
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0], str(args.folder), *args.shapes, "-n", "1"]
        shape_fixer.main()
        for shape in args.shapes:
            files = list(Path(f"{args.folder}_{shape}").glob("*.tflite"))
            if len(files) != 1 or files[0].stat().st_size == 0:
                raise ValueError(f"静态图未完整生成: {shape}")
        Path("shape_adapter.json").write_text(json.dumps({
            "installed_sdk_modified": False, "adapters": adapters,
            "shapes": args.shapes,
        }, indent=2) + "\n")
    finally:
        sys.argv = saved_argv
        builder.ModelBuilder.import_subgraph = original
        utils.temp_file = original_temp_file


def quantize(arguments=None):
    """直接调用 SDK 量化函数,仅在 SDK 命令行接口边界临时设置参数."""
    parser = ptq.get_argument_parser()
    parser.add_argument("--audit-only", action="store_true",
                        help="只核对已有量化图,不重新量化.")
    args = parser.parse_args(arguments)
    if args.audit_only:
        verify_weight_precision(args, utils)
        return
    source_path = Path(ptq.__file__)
    source = source_path.read_text()
    if args.precision != ["asym4W_sym16A"] or args.tail_precision not in (
            None, "asym4W_sym16A"):
        raise ValueError("本项目仅允许全 W4A16,请排查误差来源而非提高权重位宽.")
    Path("quantization_adapter.json").write_text(json.dumps({
        "sdk_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "adapter": "w4_only_strict_exceptions_weight_audit",
        "installed_sdk_modified": False,
    }, indent=2) + "\n")
    print("[SDK 量化] 当前进程直接调用 ptq.main(),转换异常将使任务失败.", flush=True)
    saved_argv = sys.argv
    original_temp_file = utils.temp_file
    try:
        if arguments is not None:
            sys.argv = [saved_argv[0], *arguments]
        utils.temp_file = strict_temp_file
        ptq.main()
        verify_weight_precision(args, utils)
    finally:
        sys.argv = saved_argv
        utils.temp_file = original_temp_file


def main():
    """保留已有命令行接口,模型代码可直接导入转换函数."""
    if len(sys.argv) > 1 and sys.argv[1] == "shape":
        fix_native_shapes()
    else:
        quantize()


if __name__ == "__main__":
    main()
