# 性能评估

运行编号 `20260928_depth_da2k_full_v1` 对 1,033 张 DA-2K 图片逐图调用 `neuronrt -m hw`，CLI 墙钟平均 `194.7004188777 ms/图`。[完整报告](../results/full_accuracy/20260928_depth_da2k_full_v1/summary.json)。该数值包含每图进程启动及模型加载，不是纯 NPU 延迟或常驻模型吞吐量。

预热后的稳定延迟、吞吐和峰值内存仍待评测。三次早期 `neuronrt -m hw` 冒烟只用于验证硬件推理及重复性，见 [smoke.md](smoke.md)。后续应在常驻板端进程中区分预处理、输入拷贝、NPU 推理和输出还原，并记录预热、分位数和内存。
