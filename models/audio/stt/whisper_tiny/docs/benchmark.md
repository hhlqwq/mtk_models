# 性能

历史板端运行 `20260928_whisper_testclean_full_v1` 共处理 2,620 条音频。Encoder 与 Decoder 的 Neuron Runtime 调用总耗时平均 **460.05 ms/条**；平均 NPU RTF 为 `0.07149`。[原始报告](../results/full_accuracy/20260928_whisper_testclean_full_v1/summary.json)。

计时不包含模型加载、FLAC 解码、Log-Mel 和文本解码。长于 30 秒的输入使用固定窗口。当前单脚本流程尚未重新实测。
