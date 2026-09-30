# YOLO-World XL 精度

COCO val2017 全量 5000 张图片，统一 80 类映射、分数阈值 0.001、NMS IoU 0.65、每图最多 300 框。目标检测只展示主指标 mAP@0.5:0.95。

| 后端 | mAP@0.5:0.95 |
| --- | ---: |
| FP32 ONNX | 0.472953 |
| 开发板混合 Neuron/CPU EP | 0.472952 |

该板端会话包含 CPU 回退，不能视为纯 NPU 精度通过。结果见[参考报告](../results/reference_accuracy/coco_fp32_v1/summary.json)和[板端报告](../results/full_accuracy/20260928_yoloworld_coco_full_v3/summary.json)。纯 NPU 配置仍有分类输出数值漂移，正式交付状态保持板端已验证。
