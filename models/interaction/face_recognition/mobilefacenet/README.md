# MobileFaceNet

人脸特征提取与验证，输入为对齐后的 112×112 RGB 人脸。模型不包含人脸检测、活体检测或开放集拒识。来源见[模型卡](model_card.md)，历史结果见[精度报告](docs/accuracy.md)和[性能报告](docs/benchmark.md)。

## 运行

1. 在 [run.sh](deploy/run.sh) 中配置权重、对齐人脸校准集、模型输出目录、板端 LFW 数据集目录、编译环境、板端地址和部署目录。
2. 在编译主机的本模型目录执行 `bash deploy/run.sh`。脚本在 Docker 中转换并编译 DLA，在主机交叉编译 C++ 程序并上传所需文件。
3. 登录所配置的开发板，执行脚本打印的 `bash .../run.sh` 命令；结果写入 `BOARD_RESULTS_DIR`。

LFW 全量验证使用 6000 对、10 折。历史非对齐输入的准确率无效，正确协议的准确率仍待重测；修改后的脚本尚未重新实测。
