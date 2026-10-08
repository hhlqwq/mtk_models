# YOLOv8n 模型卡

- 任务: COCO 80 类目标检测.
- 平台: Genio 720 / MT8189 / MDLA 5.3; Genio 5100 未验证.
- 来源: Ultralytics v8.0.111,官方 `yolov8n.pt`,AGPL-3.0.
- 输入: RGB NCHW FP32 原始导出,INT8 NPU 输入,固定 640x640.
- 适配: 六个原始检测头,CPU DFL、Sigmoid、类别内 NMS.
- 量化: 固定排序 100 张 COCO 图片,逐通道权重量化、逐张量 INT8 IO.
- 评测: COCO val2017 5000 张,三后端同协议,结果以 `results/` 为准.
- 使用限制: 来源、协议、板端环境和真实验证范围详见 README.
