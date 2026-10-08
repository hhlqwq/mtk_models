# Depth Anything V2 Small

完整测试成功后只保留一个 `summary.json`,汇总板端 NPU 平均耗时、核心精度及参考基准的精度差值。原始预测、缓存与日志在汇总成功后自动删除; 失败时保留本次 `work/` 目录。`REFERENCE_ACCURACY` 与 `REFERENCE_SOURCE` 在脚本顶部配置,历史参考会明确标注; 缺少匹配基准时不计算差值。

运行时的临时文件与 C++ 程序保存在脚本顶部的 `BUILD_WORK_DIR`,默认是仓库外的 `/tmp/hailongcodex/<当天日期>/depth_anything_v2_small/`。模型产物写入 `MODEL_OUTPUT_DIR`,板端测试结果写入 `BOARD_RESULTS_DIR`。脚本不在代码目录生成 Python 字节码缓存。

单目相对深度估计，输入为 518×518 RGB 图像。输出只表达单张图内的相对结构，不能直接用于米制测距。来源见[模型卡](model_card.md)，历史结果见[精度报告](docs/accuracy.md)和[性能报告](docs/benchmark.md)。

## 运行

1. 在 [run.sh](deploy/run.sh) 中配置原始权重、上游源码、校准图片、DA-2K 数据集、模型输出目录、Docker 与交叉编译环境、板端地址和部署目录。
2. 在编译主机的本模型目录执行 `bash deploy/run.sh`。脚本在 Docker 中转换并编译 DLA，在主机交叉编译板端 C++ 程序并上传。
3. 登录所配置的开发板，执行脚本打印的 `bash .../run.sh` 命令；结果写入 `BOARD_RESULTS_DIR`。

历史 DA-2K 全量点对准确率为 ONNX 94.83%、板端 85.78%。修改后的脚本尚未重新实测。
