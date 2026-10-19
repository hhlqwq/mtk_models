"""读取板端 JSON 配置,完成本模型推理与原始指标计算."""

from tools.board import command, context
from tools.evaluation import is_test


def main():
    """执行当前运行,快速结果仅在 results 下追加 _test 后缀."""
    ctx = context(__file__)
    if ctx.result.exists():
        raise FileExistsError(f"结果目录已存在: {ctx.result}.")
    print(f"[板端] 结果目录: {ctx.result}", flush=True)
    (ctx.work / "report").mkdir(parents=True)
    detections = ctx.dataset / "person_detection_results/COCO_val2017_detections_AP_H_56_person.json"
    command(ctx, "prepare_eval_manifest", "--detections", detections,
            "--output", ctx.work / "person_detections.tsv")
    command(ctx, "rtmpose_board_eval", "--model", ctx.root / "models/model_int8.dla",
            "--images", ctx.dataset / "images", "--manifest", ctx.work / "person_detections.tsv",
            "--output-dir", ctx.work, "--warmup", "20", "--progress-interval", "500")
    command(ctx, "evaluate_coco_wholebody.py", "--annotations",
            ctx.dataset / "annotations/coco_wholebody_val_v1.0.json",
            "--predictions", ctx.work / "predictions.jsonl",
            "--formatted", ctx.work / "wholebody_predictions.json",
            "--metrics", ctx.work / "coco_wholebody_metrics.json",
            "--summary-log", ctx.work / "coco_wholebody_summary.log")


if __name__ == "__main__":
    main()
