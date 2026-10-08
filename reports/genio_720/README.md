# Genio 720 当前测试结果

只展示用户上传的模型 `results/summary.json`,完整模型列表见 [模型总览](../../models/README.md)。未上传的模型待补充实测结果。

| 模型 | NPU 平均耗时 (ms) | 峰值 RSS (MiB) | ONNX FP32 核心精度 | 板端核心精度 | 精度下降 (百分点) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [YOLOv5s](../../models/perception/object_detection/yolov5s/README.md) | 9.736857 | 未记录 | 0.370900 | 0.3585986035 | 1.230140 |

YOLOv5s 使用 COCO val2017 全量 5000 张图片,指标为 mAP@0.5:0.95。运行编号为 `20261008_024751_55865`,FP32 为已确认的同协议 ONNX 基准。原始当前汇总见 [summary.json](../../models/perception/object_detection/yolov5s/results/summary.json)。
