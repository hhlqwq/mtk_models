# YOLO-World XL 精度

COCO val2017 全量 5000 张图片，统一 80 类映射、分数阈值 0.001、NMS IoU 0.65、每图最多 300 框。目标检测只展示主指标 mAP@0.5:0.95。

| 后端 | mAP@0.5:0.95 |
| --- | ---: |
| FP32 ONNX | 0.472953 |
| 开发板混合 Neuron/CPU EP | 0.472952 |

该历史板端会话包含 CPU 回退，不能视为纯 NPU 精度。结果见[参考报告](../results/reference_accuracy/coco_fp32_v1/summary.json)和[板端报告](../results/full_accuracy/20260928_yoloworld_coco_full_v3/summary.json)。

改写后的 ONNX 在三张公开图片上完成 FP32 与 Genio 720 纯 Neuron 对照。板端 C++ 对原始 DFL logits 做 CPU 后处理，三图检测数量分别为 10、13、4；类别逐项一致，最大分数差 0.00535、最大坐标差 0.156 像素。[三图比较](../examples/output/public/comparison.json)及[可视化](../examples/output/README.md)已保存。三图不构成 COCO 全量 mAP，纯 NPU 路径的全量精度仍待运行。
