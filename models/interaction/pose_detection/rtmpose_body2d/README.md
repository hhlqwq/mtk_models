# RTMPose Body2d

完整测试成功后只保留一个 `summary.json`。预测、缓存与日志在汇总成功后删除,失败时保留本次 `work/`。参考基准默认留空,填写已确认的同协议基准后计算精度差值。

运行时的临时文件与 C++ 程序保存在脚本顶部的 `BUILD_WORK_DIR`,默认是仓库外的 `/tmp/hailongcodex/<当天日期>/rtmpose_body2d/`。模型产物写入 `MODEL_OUTPUT_DIR`,板端测试结果写入 `BOARD_RESULTS_DIR`。脚本不在代码目录生成 Python 字节码缓存。

人体 WholeBody 姿态估计，输入为 256×192 RGB 人体裁剪图，输出 133 个关节点。正式上游为 OpenMMLab MMPose v1.3.2；来源见[模型卡](model_card.md)。

## 运行

1. 在 [run.sh](deploy/run.sh) 中配置 ONNX、校准图片及标注、模型输出目录、COCO-WholeBody 数据集、Docker 与交叉编译环境、板端地址和部署目录。
2. 在编译主机的本模型目录执行 `bash deploy/run.sh`。脚本在 Docker 中量化和编译 DLA，在主机交叉编译板端 C++ 程序并上传。
3. 登录所配置的开发板，执行脚本打印的 `bash .../run.sh` 命令。板端处理 5000 张图片和 104125 个人体框，结果写入 `BOARD_RESULTS_DIR`。

## 当前测试结果

待上传本模型的 `results/summary.json` 后更新。只记录板端 NPU 平均耗时、推理进程峰值 RSS (MiB)、任务核心精度及同协议 ONNX 参考精度差值。峰值 RSS 包含运行库及前后处理,不代表 NPU 专用内存。
