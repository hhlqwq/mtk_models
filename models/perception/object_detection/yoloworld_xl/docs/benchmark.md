# 性能

历史板端运行 `20260928_yoloworld_coco_full_v3` 处理 5,000 张图片，常驻 ONNX Runtime 会话的 `session.Run` 平均耗时 **3864.08 ms/张**。[原始报告](../results/full_accuracy/20260928_yoloworld_coco_full_v3/summary.json)。

该会话同时执行 Neuron EP 与 CPU 节点，计时不包含逐图预处理和后处理，不能作为纯 NPU 耗时。当前单脚本流程尚未重新实测。
