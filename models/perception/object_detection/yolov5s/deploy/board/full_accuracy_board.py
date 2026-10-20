"""读取本次 C++ 检测输出,按原有 COCO 协议计算精度."""

from tools.board import command, context
from tools.evaluation import is_test


def main():
    """使用与推理相同的配置和运行编号,保留同范围浮点报告匹配."""
    ctx = context(__file__)
    command(ctx, "evaluate_coco.py",
            "--annotations", ctx.dataset / "annotations/instances_val2017.json",
            "--predictions", ctx.work / "predictions.json",
            "--processed-ids", ctx.work / "processed_ids.txt",
            "--timings", ctx.work / "timing_summary_current_run.json",
            "--metrics", ctx.result / "summary.json", "--run-id", ctx.run_id,
            "--fp32-report", ctx.root / "models" / (
                "onnx_reference_test.json" if is_test() else "onnx_reference.json"))


if __name__ == "__main__":
    main()
