# Whisper-Tiny

完整测试成功后只保留一个 `summary.json`,汇总板端 NPU 平均耗时、核心精度及参考基准的精度差值。原始预测、缓存与日志在汇总成功后自动删除; 失败时保留本次 `work/` 目录。`REFERENCE_ACCURACY` 与 `REFERENCE_SOURCE` 在脚本顶部配置,历史参考会明确标注; 缺少匹配基准时不计算差值。WER 越低越好,差值按板端 WER 减参考 WER 计算; 默认参考是历史 FP16,本模型不是 INT8 量化流程。

运行时的临时文件与 C++ 程序保存在脚本顶部的 `BUILD_WORK_DIR`,默认是仓库外的 `/tmp/hailongcodex/<当天日期>/whisper_tiny/`。模型产物写入 `MODEL_OUTPUT_DIR`,板端测试结果写入 `BOARD_RESULTS_DIR`。脚本不在代码目录生成 Python 字节码缓存。

多语言语音识别。输入为 16 kHz 单声道音频，板端运行 Encoder 和 Decoder 两个 DLA。模型来源见[模型卡](model_card.md)，历史结果见[精度报告](docs/accuracy.md)和[性能报告](docs/benchmark.md)。

## 运行

1. 在 [run.sh](deploy/run.sh) 中配置 Encoder、Decoder ONNX，模型输出目录，LibriSpeech `test-clean` 数据集，Docker 容器、交叉编译工具链，以及 `BOARD_HOST`、`BOARD_DEPLOY_DIR`。
2. 在编译主机的本模型目录执行 `bash deploy/run.sh`。脚本在 Docker 中转换并编译双 DLA，在主机交叉编译板端 C++ 程序，然后上传所需文件。
3. 登录所配置的开发板，执行脚本打印的 `bash .../run.sh` 命令。板端准备 2620 条音频、运行 C++ 推理并计算 WER；结果写入 `BOARD_RESULTS_DIR`。

模型和数据由用户放在配置的目录，脚本不下载大文件。历史 `test-clean` 全量结果为参考端 WER 7.5546%、板端 WER 7.5603%；修改后的脚本尚未重新实测。
