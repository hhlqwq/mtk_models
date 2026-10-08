# MobileFaceNet

完整测试成功后只保留一个 `summary.json`,汇总板端 NPU 平均耗时、核心精度及参考基准的精度差值。原始预测、缓存与日志在汇总成功后自动删除; 失败时保留本次 `work/` 目录。`REFERENCE_ACCURACY` 与 `REFERENCE_SOURCE` 在脚本顶部配置,历史参考会明确标注; 缺少匹配基准时不计算差值。

运行时的临时文件与 C++ 程序保存在脚本顶部的 `BUILD_WORK_DIR`,默认是仓库外的 `/tmp/hailongcodex/<当天日期>/mobilefacenet/`。模型产物写入 `MODEL_OUTPUT_DIR`,板端测试结果写入 `BOARD_RESULTS_DIR`。脚本不在代码目录生成 Python 字节码缓存。

人脸特征提取与验证，输入为对齐后的 112×112 RGB 人脸。模型不包含人脸检测、活体检测或开放集拒识。来源见[模型卡](model_card.md)，历史结果见[精度报告](docs/accuracy.md)和[性能报告](docs/benchmark.md)。

## 运行

1. 在 [run.sh](deploy/run.sh) 中配置权重、对齐人脸校准集、模型输出目录、板端 LFW 数据集目录、编译环境、板端地址和部署目录。
2. 在编译主机的本模型目录执行 `bash deploy/run.sh`。脚本在 Docker 中转换并编译 DLA，在主机交叉编译 C++ 程序并上传所需文件。
3. 登录所配置的开发板，执行脚本打印的 `bash .../run.sh` 命令；结果写入 `BOARD_RESULTS_DIR`。

LFW 全量验证使用 6000 对、10 折。历史非对齐输入的准确率无效，正确协议的准确率仍待重测；修改后的脚本尚未重新实测。
