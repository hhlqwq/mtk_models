"""读取板端配置并调度 C++ 推理,不计算精度指标."""

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
            "--output-dir", ctx.work, "--warmup", "20" if is_test() else "0", "--progress-interval", "500")



if __name__ == "__main__":
    main()
