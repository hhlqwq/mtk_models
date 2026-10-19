"""导出原生 Prefill 和 Decode 的静态单分片模型."""

from tools.deploy import arguments
from tools.llm.native import python, settings


def main():
    """保留原有静态形状与单分片规则."""
    native = settings(arguments(__file__))
    python(native, native.ctx.repo / "tools/llm/quantize_native.py", "shape",
           native.quantized, f"{native.prefill}t{native.context}c", f"1t{native.context}c", "-n", "1")


if __name__ == "__main__":
    main()
