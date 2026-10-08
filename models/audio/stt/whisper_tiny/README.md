# Whisper-Tiny

完整测试成功后只保留一个 `summary.json`。预测、缓存与日志在汇总成功后删除,失败时保留本次 `work/`。参考基准默认留空,填写已确认的同协议基准后计算精度差值。

运行时的临时文件与 C++ 程序保存在脚本顶部的 `BUILD_WORK_DIR`,默认是仓库外的 `/tmp/hailongcodex/<当天日期>/whisper_tiny/`。模型产物写入 `MODEL_OUTPUT_DIR`,板端测试结果写入 `BOARD_RESULTS_DIR`。脚本不在代码目录生成 Python 字节码缓存。

多语言语音识别。输入为 16 kHz 单声道音频，板端运行 Encoder 和 Decoder 两个 DLA。模型来源见[模型卡](model_card.md)。

## 运行

1. 在 [run.sh](deploy/run.sh) 中配置 Encoder、Decoder ONNX，模型输出目录，LibriSpeech `test-clean` 数据集，Docker 容器、交叉编译工具链，以及 `BOARD_HOST`、`BOARD_DEPLOY_DIR`。
2. 在编译主机的本模型目录执行 `bash deploy/run.sh`。脚本在 Docker 中转换并编译双 DLA，在主机交叉编译板端 C++ 程序，然后上传所需文件。
3. 登录所配置的开发板，执行脚本打印的 `bash .../run.sh` 命令。板端准备 2620 条音频、运行 C++ 推理并计算 WER；结果写入 `BOARD_RESULTS_DIR`。

模型和数据由用户准备,脚本不下载大文件.

## 当前测试结果

待上传本模型的 `results/summary.json` 后更新。只记录板端 NPU 平均耗时、推理进程峰值 RSS (MiB)、任务核心精度及同协议 ONNX 参考精度差值。峰值 RSS 包含运行库及前后处理,不代表 NPU 专用内存。

主指标 WER 越低越好,差值为板端 WER 减参考 WER; 参考后端须如实记录。
