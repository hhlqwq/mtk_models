# YOLO-World XL 模型卡

## 来源与许可证

- 算法上游：<https://github.com/AILab-CVC/YOLO-World>.
- 固定提交：`4f70adbaacf5685bd9ec5bea85f1f91057f6fc0b`.
- 上游许可证：GPL-3.0,许可证原文保存在 `LICENSE`.
- 部署资产：MediaTek IoT AI Hub Model Zoo 的 `yoloworld_xl.onnx`.

MediaTek 文件没有提供上游 checkpoint 名称、导出配置或内置文本清单.根据固定输入、六个
输出形状和 80 个类别通道,可确认它是重参数化的 80 类三尺度检测图；结合 YOLO-World
上游默认 ONNX 导出行为,本项目按 COCO 80 类顺序解释输出,但把"MediaTek 二进制对应的
精确上游权重"保留为未公开,而不是推测填写.

## 兼容修改

- 官方模型：ONNX IR 6、opset 11、513 个节点、232 个 initializer.
- 兼容问题：三个 `axis=3` Softmax 被板端 Neuron EP 以 `SinceVersion() < 13` 拒绝.
- 修改方法：ONNX version converter 将完整模型转换为 opset 13.
- 等价门槛：ONNX checker 通过,六个 CPU EP 输出逐元素完全一致.
- 纯 NPU 修改：从 Raw ONNX 将四处固定文本注意力和三处分类矩阵乘法改写为卷积，将八处通道 Split 改写为 Slice，并将注意力门控改为四维池化与卷积。DFL 留在 C++ CPU 后处理。

## 目标平台

- Genio 720 / MT8189.
- Rity Demo 26.0-dev、Linux 6.6.117.
- ONNX Runtime 1.20.2.
- `NeuronExecutionProvider`,FP32 输入图通过强制选项在 NPU 上以 FP16 执行.

## 限制

- 当前模型只能使用导出时固化的 80 类文本,板端不能动态输入任意开放词汇.
- Neuron EP 允许不支持的节点回退 CPU,板端 C++ 在正式处理图片前检查 profiling；发现 CPU 模型节点即停止。
- 官方参考性能不能代替本项目板端实测.

## 当前测试结果

COCO val2017 全量 5000 张.同协议 ONNX FP32 mAP@0.5:0.95 为 47.29%,板端为 47.27%,下降 0.03 个百分点.板端 ORT 平均调用耗时为 303.42 ms,独立 NPU 耗时未采集; 推理进程峰值 RSS 为 1459.28 MiB.

结果见 [summary.json](results/summary.json),核心指标及三张实际检测效果见 [README](README.md#当前测试结果).
