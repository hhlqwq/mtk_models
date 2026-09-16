# 精度报告

## 当前状态

正式 COCO bbox mAP 尚未执行,不能用三张公开样例替代.

兼容模型必须先通过 `deploy/verify_onnx_equivalence.py`,证明 opset 11 到 opset 13 转换
没有改变六个 CPU EP 输出.随后通过真实图片对比 CPU EP 与 Neuron EP 的检测框、类别和
置信度,记录 FP16 硬件执行带来的差异.

正式精度计划使用板端已有 COCO val2017 5,000 张图片及 `instances_val2017.json`,固定
score threshold、IoU threshold、最大检测数和 COCO 80 类映射.只有完整处理清单、预测
JSON、模型/数据集/代码哈希和 pycocotools 指标齐全后,才更新本报告.
