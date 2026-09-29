# FastSAM-s 模型卡

| 项目 | 配置 |
| --- | --- |
| 任务 | 类别无关实例分割 |
| 模型来源 | CASIA-LMC-Lab/FastSAM 官方 FastSAM-s.pt |
| 官方源码参考提交 | b4ed20c2fed75eadc5aa7d8b09fedd137b873b52 |
| 实际导出实现 | 容器预装 Ultralytics 8.0.111,已验证真实权重前向和 ONNX 导出 |
| 权重 SHA-256 | `e9034d7478a8e9d1bfb57b51592e521a253287c7cdcf79258f61ea6d68584a0d`,本次文件实测值 |
| 输入 | RGB, float32, NCHW, [1,3,640,640], /255 |
| 缩放 | 保持比例,居中 letterbox,填充值 114 |
| NPU 输出 | stride 8/16/32 的 64 通道框 logits、1 通道分数 logits、32 通道掩码系数,以及 [1,32,160,160] 原型 |
| 板端实现 | C++ Neuron Runtime API + OpenCV |
| CPU 后处理 | C++ DFL、Sigmoid、类别无关 NMS、原型线性组合、原图掩码还原 |
| Demo 默认阈值 | confidence=0.4, NMS IoU=0.9, max_det=100；正式 AR 评测 confidence=0.001 |
| 提示支持 | 全图、单前景点、单 xyxy 框; 提示语义见 Demo 文档 |
| 尚不支持 | 文本/CLIP、负点组合、多图批处理、G5100 实测 |
| 精度状态 | PyTorch / ONNX 全量 segm AR@100 为 0.391 / 0.390；旧 INT8 的 0.370 使用了重叠校准图片，仅作历史诊断；独立校准模型全量复评中 |
| 交付状态 | 板端已验证；旧 DLA 常驻性能已测，独立校准 DLA 全量精度与性能复评中 |

FastSAM 不输出 COCO 80 类语义标签.本次正式精度采用类别无关协议,
不能把所有预测随意设置为某个 COCO 类别后宣称完成标准实例分割 mAP.
640 输入是本项目端侧配置,不直接比较官方 1024 输入结果.

记录和哈希由 `export_manifest.json`、`model_int8.json`、`SHA256SUMS`、
`DLA_SHA256SUMS` 和板端 `results.json` 承载.本次证据见
[精度报告](docs/accuracy.md)与[2026-09-23 板端冒烟报告](docs/board_smoke_20260923.md).
