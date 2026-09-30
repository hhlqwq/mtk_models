# FastSAM-s

类别无关实例分割，输入为 640×640 RGB 图像。来源见[模型卡](model_card.md)，历史结果见[精度报告](docs/accuracy.md)和[性能报告](docs/benchmark.md)。

## 运行

1. 在 [run.sh](deploy/run.sh) 中配置原始权重、校准图、示例图、COCO val2017 数据集、模型输出目录、Docker 与交叉编译环境、板端地址和部署目录。
2. 在编译主机的本模型目录执行 `bash deploy/run.sh`。脚本在 Docker 中转换并编译 DLA，在主机交叉编译 C++ 程序并上传。
3. 登录所配置的开发板，执行脚本打印的 `bash .../run.sh` 命令。板端逐图保存检查点并计算 COCO 分割指标；结果写入 `BOARD_RESULTS_DIR`。

历史独立校准 DLA 的 COCO val2017 AR@100 为 0.376384，常驻 Runtime 调用均值 14.0536 ms/图。三张公开样例已补齐 [FP32 ONNX 与板端分割图](examples/output/README.md)；尾部实例有差异。修改后的单脚本流程尚未重新实测。
