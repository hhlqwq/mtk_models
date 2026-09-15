# YOLO-World XL 模型卡

## 来源与许可证

- 算法上游：<https://github.com/AILab-CVC/YOLO-World>。
- 固定提交：`4f70adbaacf5685bd9ec5bea85f1f91057f6fc0b`。
- 上游许可证：GPL-3.0，许可证原文保存在 `LICENSE`。
- 部署资产：MediaTek IoT AI Hub Model Zoo 的 `yoloworld_xl.onnx`。
- 官方 ONNX SHA-256：`6d5b231425200f0426b73967c33a69ce5af83e993426d4fc64dc1709571d6174`。

MediaTek 文件没有提供上游 checkpoint 名称、导出配置或内置文本清单。根据固定输入、六个
输出形状和 80 个类别通道，可确认它是重参数化的 80 类三尺度检测图；结合 YOLO-World
上游默认 ONNX 导出行为，本项目按 COCO 80 类顺序解释输出，但把“MediaTek 二进制对应的
精确上游权重”保留为未公开，而不是推测填写。

## 兼容修改

- 官方模型：ONNX IR 6、opset 11、513 个节点、232 个 initializer。
- 兼容问题：三个 `axis=3` Softmax 被板端 Neuron EP 以 `SinceVersion() < 13` 拒绝。
- 修改方法：ONNX version converter 将完整模型转换为 opset 13。
- 等价门槛：ONNX checker 通过，六个 CPU EP 输出逐元素完全一致。

## 目标平台

- Genio 720 / MT8189。
- Rity Demo 26.0-dev、Linux 6.6.117。
- ONNX Runtime 1.20.2。
- `NeuronExecutionProvider`，FP32 输入图通过强制选项在 NPU 上以 FP16 执行。

## 限制

- 当前模型只能使用导出时固化的 80 类文本，板端不能动态输入任意开放词汇。
- Neuron EP 允许不支持的节点回退 CPU，因此性能报告必须附 profiling 节点归属。
- 官方参考性能不能代替本项目板端实测。
- 正式 COCO 精度尚未完成时，交付状态最高为“板端已验证”。
