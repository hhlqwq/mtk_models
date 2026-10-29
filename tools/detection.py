"""检测模型共用的 Python 板端推理循环,模型解码由调用者提供."""

import json
import time

import cv2
import numpy as np

from tools.evaluation import coco_ids, is_test
from tools.runtime import Runtime, peak_rss_kib, quantize
from tools.output_metrics import OutputComparison


COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
            74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def infer(ctx, input_info, output_info, preprocess, postprocess):
    """复用单个 Runtime,保存 COCO 预测及评价阶段必要的耗时汇总."""
    if ctx.result.exists():
        raise FileExistsError(ctx.result)
    images = sorted((ctx.dataset / "images").glob("*.jpg"))
    selected = set(coco_ids(int(path.stem) for path in images))
    images = [path for path in images if int(path.stem) in selected]
    if not images:
        raise ValueError("没有找到检测输入图片.")
    ctx.work.mkdir(parents=True)
    predictions, timings = [], []
    comparison = OutputComparison(ctx)
    with Runtime(ctx.root / "model_int8.dla") as model:
        if tuple(map(len, model.sizes)) != (1, len(output_info)):
            raise ValueError("检测模型 I/O 数量不符.")
        inputs, input_view = model.tensor(0, input_info["shape"], np.int8,
                                          fill=input_info["zero_point"])
        output_buffers, output_views = [], []
        for index, item in enumerate(output_info):
            storage, view = model.tensor(index, item["shape"], np.int8, output=True)
            output_buffers.append(storage)
            output_views.append(view)
        if not is_test():
            values, _ = preprocess(cv2.imread(str(images[0])))
            np.copyto(input_view, quantize(values, input_info["scale"],
                                          input_info["zero_point"]))
            for _ in range(20):
                model.infer([inputs], output_buffers)
        for count, path in enumerate(images, 1):
            started = time.perf_counter()
            values, geometry = preprocess(cv2.imread(str(path)))
            np.copyto(input_view, quantize(values, input_info["scale"],
                                          input_info["zero_point"]))
            prepared = time.perf_counter()
            milliseconds = model.infer([inputs], output_buffers)
            inferred = time.perf_counter()
            outputs = [(view.astype(np.float32) - item["zero_point"]) * item["scale"]
                       for view, item in zip(output_views, output_info)]
            boxes, scores, classes = postprocess(outputs, geometry)
            ended = time.perf_counter()
            comparison.update([values], {item.get("name", f"head_{index}"): value
                                         for index, (item, value) in enumerate(zip(output_info, outputs))},
                              sample_id=path.name, sample_files=[path])
            for box, score, category in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(float, box)
                predictions.append({"image_id": int(path.stem),
                                    "category_id": COCO_IDS[int(category)],
                                    "bbox": [round(value, 3) for value in (x1, y1, x2 - x1, y2 - y1)],
                                    "score": round(float(score), 5)})
            if not is_test():
                timings.append([(prepared - started) * 1000, milliseconds,
                                (ended - inferred) * 1000, (ended - started) * 1000])
            if count % 50 == 0 or count == len(images):
                print(f"[推理] {ctx.config['name']} {count}/{len(images)}", flush=True)
    (ctx.work / "predictions.json").write_text(json.dumps(predictions))
    comparison.finish()
    (ctx.work / "processed_ids.txt").write_text(
        "".join(f"{int(path.stem)}\n" for path in images))
    if not is_test():
        report = {"processed_images": len(images), "peak_rss_kb": peak_rss_kib(),
                  "backend": "python_neuron_runtime_hw"}
        for index, name in enumerate(("preprocess_ms", "npu_ms", "postprocess_ms", "e2e_ms")):
            values = np.asarray(timings)[:, index]
            report[name] = {"mean": float(values.mean()),
                            "p95": float(np.percentile(values, 95))}
        (ctx.work / "timing_summary_current_run.json").write_text(json.dumps(report))
