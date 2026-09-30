# YOLO-World XL

开放词汇目标检测，当前模型固化 COCO 80 类文本嵌入。使用 MediaTek 官方 ONNX，经 ONNX Runtime Neuron EP 在板端运行。来源见[模型卡](model_card.md)，历史结果见[精度报告](docs/accuracy.md)和[性能报告](docs/benchmark.md)。

## 运行

1. 在 [run.sh](deploy/run.sh) 中配置官方 ONNX、模型输出目录、COCO val2017 数据集、交叉编译工具链、板端 ONNX Runtime 库、板端地址和部署目录。
2. 在编译主机的本模型目录执行 `bash deploy/run.sh`。脚本准备兼容 ONNX、交叉编译板端 C++ 程序并上传。
3. 登录所配置的开发板，执行脚本打印的 `bash .../run.sh` 命令；结果写入 `BOARD_RESULTS_DIR`。

历史混合 Neuron/CPU EP 的 COCO bbox AP50:95 为 0.472952，纯 NPU 正确性尚未通过。修改后的脚本尚未重新实测。
