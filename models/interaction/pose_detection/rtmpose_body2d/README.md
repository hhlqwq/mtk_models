# RTMPose Body2d

完整测试成功后只保留一个 `summary.json`,汇总板端 NPU 平均耗时、核心精度及参考基准的精度差值。原始预测、缓存与日志在汇总成功后自动删除; 失败时保留本次 `work/` 目录。`REFERENCE_ACCURACY` 与 `REFERENCE_SOURCE` 在脚本顶部配置,历史参考会明确标注; 缺少匹配基准时不计算差值。

运行时的临时文件与 C++ 程序保存在脚本顶部的 `BUILD_WORK_DIR`,默认是仓库外的 `/tmp/hailongcodex/<当天日期>/rtmpose_body2d/`。模型产物写入 `MODEL_OUTPUT_DIR`,板端测试结果写入 `BOARD_RESULTS_DIR`。脚本不在代码目录生成 Python 字节码缓存。

人体 WholeBody 姿态估计，输入为 256×192 RGB 人体裁剪图，输出 133 个关节点。正式上游为 OpenMMLab MMPose v1.3.2；来源见[模型卡](model_card.md)，历史结果见[精度报告](docs/accuracy.md)和[性能报告](docs/benchmark.md)。

## 运行

1. 在 [run.sh](deploy/run.sh) 中配置 ONNX、校准图片及标注、模型输出目录、COCO-WholeBody 数据集、Docker 与交叉编译环境、板端地址和部署目录。
2. 在编译主机的本模型目录执行 `bash deploy/run.sh`。脚本在 Docker 中量化和编译 DLA，在主机交叉编译板端 C++ 程序并上传。
3. 登录所配置的开发板，执行脚本打印的 `bash .../run.sh` 命令。板端处理 5000 张图片和 104125 个人体框，结果写入 `BOARD_RESULTS_DIR`。

历史同协议 ONNX WholeBody AP 为 0.5703，INT8 板端 AP 为 0.5324。修改后的脚本尚未重新实测。
