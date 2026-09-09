"""验证 RTMPose 原始 ONNX 与 MTK 兼容模型的数值等价性."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime


def run_model(model_path: Path, input_data: np.ndarray) -> list[np.ndarray]:
    """使用 ONNX Runtime CPU 执行 RTMPose 双输出模型."""
    session = onnxruntime.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 2:
        raise ValueError(
            f"模型 I/O 数量异常: inputs={len(inputs)}, outputs={len(outputs)}")
    return session.run(
        [output.name for output in outputs], {inputs[0].name: input_data})


def verify(args: argparse.Namespace) -> None:
    """使用固定随机输入逐输出比较降级前后数值."""
    input_data = np.random.default_rng(args.seed).random(
        (1, 3, 256, 192), dtype=np.float32)
    reference_outputs = run_model(args.reference, input_data)
    converted_outputs = run_model(args.converted, input_data)
    for index, (reference, converted) in enumerate(
            zip(reference_outputs, converted_outputs)):
        if reference.shape != converted.shape:
            raise ValueError(
                f"输出 {index} shape 不一致: "
                f"{reference.shape} != {converted.shape}")
        difference = np.abs(reference.astype(np.float64) -
                            converted.astype(np.float64))
        max_abs = float(difference.max())
        mean_abs = float(difference.mean())
        if max_abs > args.max_abs or mean_abs > args.mean_abs:
            raise ValueError(
                f"输出 {index} 数值不一致: max_abs={max_abs:.8g}, "
                f"mean_abs={mean_abs:.8g}")
        print(f"[VERIFY] 输出 {index}: max_abs={max_abs:.8g}, "
              f"mean_abs={mean_abs:.8g}")
    print("[OK] RTMPose ONNX 兼容改写数值等价.")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--max-abs", type=float, default=1e-6)
    parser.add_argument("--mean-abs", type=float, default=1e-7)
    return parser.parse_args()


if __name__ == "__main__":
    verify(parse_args())
