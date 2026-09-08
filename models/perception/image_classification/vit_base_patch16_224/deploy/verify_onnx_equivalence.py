"""验证 Qualcomm 原始 ONNX 与 MTK 兼容降级模型的数值等价性。"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime


def run_model(model_path: Path, input_data: np.ndarray) -> np.ndarray:
    """使用 ONNX Runtime CPU 执行单输入单输出模型。"""
    session = onnxruntime.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"模型 I/O 数量异常: inputs={len(inputs)}, outputs={len(outputs)}")
    return session.run([outputs[0].name], {inputs[0].name: input_data})[0]


def verify(args: argparse.Namespace) -> None:
    """使用固定随机输入比较原始模型与降级模型输出。"""
    input_data = np.random.default_rng(args.seed).random(
        (1, 3, 224, 224), dtype=np.float32)
    reference = run_model(args.reference, input_data)
    converted = run_model(args.converted, input_data)
    if reference.shape != converted.shape:
        raise ValueError(
            f"输出 shape 不一致: {reference.shape} != {converted.shape}")
    difference = np.abs(reference.astype(np.float64) -
                        converted.astype(np.float64))
    max_abs = float(difference.max())
    mean_abs = float(difference.mean())
    reference_top1 = int(reference.reshape(-1).argmax())
    converted_top1 = int(converted.reshape(-1).argmax())
    if not np.allclose(reference, converted, rtol=args.rtol, atol=args.atol):
        raise ValueError(
            f"降级前后数值不一致: max_abs={max_abs:.8g}, "
            f"mean_abs={mean_abs:.8g}")
    if reference_top1 != converted_top1:
        raise ValueError(
            f"降级前后 Top-1 不一致: {reference_top1} != {converted_top1}")
    print(f"[OK] ONNX 降级数值等价: max_abs={max_abs:.8g}, "
          f"mean_abs={mean_abs:.8g}, top1={reference_top1}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    verify(parse_args())
