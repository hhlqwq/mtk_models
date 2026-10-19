"""读取板端 JSON 配置,完成本模型推理与原始指标计算."""

from tools.board import command, context
from tools.evaluation import is_test


def main():
    """执行当前运行,快速结果仅在 results 下追加 _test 后缀."""
    ctx = context(__file__)
    if ctx.result.exists():
        raise FileExistsError(f"结果目录已存在: {ctx.result}.")
    print(f"[板端] 结果目录: {ctx.result}", flush=True)
    ctx.work.mkdir(parents=True)
    arguments = ["--model", ctx.root / "models/model_int8.dla",
                 "--images", ctx.dataset / "images", "--output-dir", ctx.work,
                 "--warmup", "20", "--confidence", "0.001", "--iou", "0.6", "--max-det", "300"]
    command(ctx, "yolov5s_board_eval", *arguments)
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
