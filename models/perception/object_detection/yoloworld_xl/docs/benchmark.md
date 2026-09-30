# 性能

历史板端运行 `20260928_yoloworld_coco_full_v3` 处理 5,000 张图片，常驻 ONNX Runtime 会话的 `session.Run` 平均耗时 **3864.08 ms/张**。[原始报告](../results/full_accuracy/20260928_yoloworld_coco_full_v3/summary.json)。

该历史会话同时执行 Neuron EP 与 CPU 模型节点，计时不包含逐图预处理和后处理，不能作为纯 NPU 耗时。

改写后的 Raw ONNX 在 Genio 720 上由常驻 C++ ONNX Runtime 会话处理三张公开样例，`session.Run` 分别为 304.097、303.176、303.538 ms/张，均值 303.604 ms/张。预热 profiling 仅有 NeuronExecutionProvider 事件，未发现 CPUExecutionProvider 模型节点。CPU DFL、框解码和 NMS 不计入该耗时；三图结果不替代正式稳态性能测试。
