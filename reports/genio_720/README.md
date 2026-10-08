# Genio 720 当前测试结果

只展示用户上传的模型 `results/summary.json`,完整模型列表见 [模型总览](../../models/README.md)。未上传的模型待补充实测结果。

| 模型 | NPU 平均耗时 (ms) | 峰值 RSS (MiB) | ONNX FP32 核心精度 | 板端核心精度 | 精度变化 (百分点) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [YOLOv5s](../../models/perception/object_detection/yolov5s/README.md) | 9.76 | 33.23 | 37.09% | 35.86% | -1.23 |

YOLOv5s 使用 COCO val2017 全量 5000 张图片,指标为 mAP@0.5:0.95。运行编号为 `20261008_032807_67603`,ONNX FP32 与板端精度均为本次同协议实测。原始当前汇总见 [summary.json](../../models/perception/object_detection/yolov5s/results/summary.json)。

YOLOv5s 采用 INT8 训练后量化 (PTQ),使用 100 张图片校准.
