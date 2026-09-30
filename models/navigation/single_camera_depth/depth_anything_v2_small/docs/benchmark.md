# 性能

历史板端常驻模型的 `NeuronRuntime_inference` API 调用平均 **133.90 ms/张**，不包含图片解码、预处理、模型加载和输出还原。

另一次逐图启动进程处理 1,033 张 DA-2K 图片，CLI 墙钟平均 `194.70 ms/张`，包含进程启动和模型加载。[原始报告](../results/full_accuracy/20260928_depth_da2k_full_v1/summary.json)。当前单脚本流程尚未重新实测。
