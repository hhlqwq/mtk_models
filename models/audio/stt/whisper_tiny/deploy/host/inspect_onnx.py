"""检查 Whisper ONNX 的静态 Shape、dtype、opset 和算子清单."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


def tensor_shape(value_info: Any) -> list[int | str]:
    """读取 ONNX ValueInfo Shape.

    Args:
        value_info: ONNX ValueInfoProto.

    Returns:
        各维度的整数值或动态维度名称.
    """
    dimensions = value_info.type.tensor_type.shape.dim
    return [
        dimension.dim_value if dimension.HasField("dim_value")
        else dimension.dim_param or "dynamic" for dimension in dimensions
    ]


def inspect_model(path: Path) -> dict[str, Any]:
    """校验单个 ONNX 并返回机器可读摘要.

    Args:
        path: ONNX 文件路径.

    Returns:
        图版本、I/O 和算子统计.

    Raises:
        ValueError: 图含动态 Shape.
    """
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    initializer_names = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input
              if item.name not in initializer_names]
    io_items = inputs + list(model.graph.output)
    dynamic = [item.name for item in io_items
               if any(not isinstance(dim, int) or dim <= 0
                      for dim in tensor_shape(item))]
    if dynamic:
        raise ValueError(f"{path.name} 存在动态或非法 Shape: {dynamic}")
    return {
        "file": path.name,
        "ir_version": model.ir_version,
        "opsets": {item.domain or "ai.onnx": item.version
                    for item in model.opset_import},
        "inputs": {item.name: tensor_shape(item) for item in inputs},
        "outputs": {item.name: tensor_shape(item)
                    for item in model.graph.output},
        "operators": dict(sorted(Counter(
            node.op_type for node in model.graph.node).items())),
    }


def main(paths: list[Path]) -> None:
    """检查所有图并写入同目录 Manifest.

    Args:
        paths: 待检查的 ONNX 路径列表.
    """
    summaries = []
    for index, path in enumerate(paths, start=1):
        print(f"[CHECK] {index}/{len(paths)}: {path}")
        summaries.append(inspect_model(path))
    output = paths[0].parent / "onnx_manifest.json"
    output.write_text(json.dumps(
        {"models": summaries}, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"[OK] ONNX 图检查通过: {output}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数.

    Returns:
        命令行参数.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="+")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().models)
