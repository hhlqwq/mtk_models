"""MTK 转换后逐张量整数 I/O 元数据的读取与校验."""

import math


def tensor_metadata(detail, dtype_name="int8", *, include_name=True):
    """校验逐张量有符号整数类型和量化参数,可保留旧格式的无名称输出."""
    if dtype_name not in ("int8", "int16", "int32"):
        raise ValueError(f"不支持的整数类型: {dtype_name}")
    quantization = detail["quantization"]
    scales, zeros = quantization["scales"], quantization["zero_points"]
    if len(scales) != 1 or len(zeros) != 1:
        raise ValueError(f"不支持的 IO 量化参数: {detail}.")
    scale = float(scales[0])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"量化尺度必须为有限正数: {detail}.")
    dtype = str(detail.get("dtype", detail.get("type", ""))).lower()
    if dtype_name not in dtype or f"u{dtype_name}" in dtype:
        raise ValueError(f"要求 {dtype_name} IO,请核实 TFLiteParser 类型字段: {detail}.")
    metadata = {"shape": [int(value) for value in detail["shape"]],
                "dtype": dtype_name, "scale": scale, "zero_point": int(zeros[0])}
    if include_name:
        metadata = {"name": str(detail["name"]), **metadata}
    return metadata
