# 性能报告

状态：全量逐图 CLI 耗时已记录；纯 NPU 延迟和内存仍待评测。

运行编号 `20260928_mobilefacenet_lfw_full_v1`，7,701 张不同图片逐图调用 `neuronrt`，CLI 墙钟平均 `36.2141417901 ms/图`。[完整报告](../results/full_accuracy/20260928_mobilefacenet_lfw_full_v1/summary.json)。该口径包含每图进程启动和模型加载，不是常驻实例的单次 NPU 推理耗时。

2026-09-24 的[板端冒烟](smoke.md)仅执行三次 `neuronrt -m hw` 推理，
未实施预热、重复计时或内存采样，因此不把冒烟耗时报告为正式性能或 FPS。
后续应分别统计检测、对齐、特征提取、匹配与多帧决策耗时。
