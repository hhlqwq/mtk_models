# Genio 720 板端验证规范

## 设备基线

```text
设备: MediaTek Genio 720 EVK
SoC: MT8189 / MT8391
系统: Rity Demo 26.0-dev (scarthgap)
地址: root@192.168.0.92
测试目录: /root/hailong.he
Neuron Runtime: 8.2.16
```

## 性能口径

每个模型至少报告：

1. `neuronrt` 预热后的纯 NPU 平均延迟。
2. Demo 预处理、NPU、后处理合计的端到端延迟。
3. 推理进程峰值常驻内存。
4. 重复次数、性能模式、输入形状、数据类型和批量大小。

延迟测试必须使用真实输入并至少进行预热。MTK 官网数据只能作为参考，不能写入“本项目实测”。

## 精度口径

同一份验证集、同一份预处理和同一套后处理分别评测：

- 原始 PyTorch 或来源框架模型。
- FP32/FP16 ONNX 模型。
- MTK NPU DLA 模型。

报告中同时给出绝对指标和相对原始模型的精度变化。
