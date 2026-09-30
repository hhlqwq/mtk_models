# 性能

历史板端常驻模型的 `NeuronRuntime_inference` API 调用平均 **0.476 ms/张**。该计时不包含检测、对齐、特征匹配和多帧决策，也不代表完整业务延迟。

另一次逐图启动进程处理 7,701 张 LFW 图片，CLI 墙钟平均 `36.21 ms/张`，包含进程启动和模型加载。[原始报告](../results/full_accuracy/20260928_mobilefacenet_lfw_full_v1/summary.json)。当前单脚本流程尚未重新实测。
