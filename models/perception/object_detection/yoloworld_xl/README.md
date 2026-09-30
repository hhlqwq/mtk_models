# YOLO-World XL

开放词汇目标检测，当前模型固化 COCO 80 类文本嵌入。使用 MediaTek 官方 ONNX，经 ONNX Runtime Neuron EP 在板端运行。来源见[模型卡](model_card.md)，精度与性能见[精度报告](docs/accuracy.md)和[性能报告](docs/benchmark.md)。

## 运行

1. 在 [run.sh](deploy/run.sh) 中配置官方 ONNX、模型输出目录、COCO val2017 数据集、交叉编译工具链、板端 ONNX Runtime 库、板端地址和部署目录。
2. 在编译主机的本模型目录执行 `bash deploy/run.sh`。脚本生成纯 NPU ONNX、交叉编译板端 C++ 程序并上传。
3. 登录所配置的开发板，执行脚本打印的 `bash .../run.sh` 命令；结果写入 `BOARD_RESULTS_DIR`。

ONNX 图将固定文本注意力和分类矩阵乘法改写为卷积，将通道 Split 改写为 Slice，并在板端完成纯 Neuron 推理。模型输出三尺度 64 通道 DFL logits，C++ 在 CPU 后处理阶段完成 DFL、框解码和 NMS。三张公开样例的 FP32 与板端检测数量均为 10、13、4，逐框最大分数差 0.00535、最大坐标差 0.156 像素；板端 profiling 无 CPU 模型节点。[六张可视化](examples/output/README.md)保存在原有示例目录。

历史混合 Neuron/CPU EP 的 COCO val2017 全量 bbox AP50:95 为 0.472952。新的纯 NPU 路径只完成三图验证，尚未运行全量精度评测。
