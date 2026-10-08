# MobileFaceNet

完整测试成功后只保留一个 `summary.json`。预测、缓存与日志在汇总成功后删除,失败时保留本次 `work/`。参考基准默认留空,填写已确认的同协议基准后计算精度差值。

运行时的临时文件与 C++ 程序保存在脚本顶部的 `BUILD_WORK_DIR`,默认是仓库外的 `/tmp/hailongcodex/<当天日期>/mobilefacenet/`。模型产物写入 `MODEL_OUTPUT_DIR`,板端测试结果写入 `BOARD_RESULTS_DIR`。脚本不在代码目录生成 Python 字节码缓存。

人脸特征提取与验证，输入为对齐后的 112×112 RGB 人脸。模型不包含人脸检测、活体检测或开放集拒识。来源见[模型卡](model_card.md)。

## 运行

1. 在 [run.sh](deploy/run.sh) 中配置权重、对齐人脸校准集、模型输出目录、板端 LFW 数据集目录、编译环境、板端地址和部署目录。
2. 在编译主机的本模型目录执行 `bash deploy/run.sh`。脚本在 Docker 中转换并编译 DLA，在主机交叉编译 C++ 程序并上传所需文件。
3. 登录所配置的开发板，执行脚本打印的 `bash .../run.sh` 命令；结果写入 `BOARD_RESULTS_DIR`。

LFW 全量验证使用 6000 对、10 折.

## 当前测试结果

待上传本模型的 `results/summary.json` 后更新。只记录板端 NPU 平均耗时、推理进程峰值 RSS (MiB)、任务核心精度及同协议 ONNX 参考精度差值。峰值 RSS 包含运行库及前后处理,不代表 NPU 专用内存。
