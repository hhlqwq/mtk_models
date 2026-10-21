# YAMNet / Genio 720

Google YAMNet 环境声音分类模型,输出 AudioSet 521 类分数.默认使用全 W8A16:权重 INT8、激活及原生 I/O 为 INT16.输入为 16 kHz 单声道音频生成的 `1×1×96×64` Log-Mel 窗口,CPU 处理音频,C++ Neuron Runtime 执行 MDLA 推理.

来源和许可证见 [模型卡](model_card.md)与[资源记录](models/source_url.txt).官方权重、源码和 ESC-50 数据集须提前放在 NAS,不在运行时下载.

## 执行范围

先在已有编译容器执行 `bash deploy/compile.sh`,完成转换、量化、编译与部署,随后自动调用板端 `compile.sh` 做少量通路检查并保存 README 示例到 `examples/output/`.该阶段不计算或发布精度、性能和峰值 RSS 报告.

正式评价使用独立的 `evaluate.sh`: 容器端只计算已有浮点模型的同协议参考及必要评价输入,资源单独放入已有部署包后,板端运行 `bash evaluate.sh`,统计精度、性能和推理进程峰值 RSS.正式结果只写入 `results/summary.json`,不重新量化或编译.预测、日志及内部指标位于当天日期临时目录.详见[板端规范](../../../../docs/genio_720.md).

下文已归档结果保留原始模型版本和评价范围,本次入口调整不代表重新实测.

## 配置与运行

模型与数据路径、量化精度在 [config.yaml](config.yaml) 中设置,公共工具链由编译容器加载.转换依赖 TensorFlow CPU 2.15.1、tf2onnx 1.16.1 等专用环境,使用平台配置中的 `MTK_YAMNET_PYTHON`;板端需要 ffmpeg 和 NumPy.

在仓库根目录进入容器后,执行:

```bash
bash models/audio/ambient_sound/yamnet/deploy/compile.sh
```

入口完成导出、PTQ、DLA 编译、C++ 构建及上传,自动生成板端示例.正式评价前按[板端规范](../../../../docs/genio_720.md)准备同协议参考,再在板端部署目录执行:

```bash
bash evaluate.sh
```

compile.sh 使用固定小子集检查通路并保存示例,不生成指标报告.默认 `conversion.precision=w8a16`,可选 `int8_tail16` 为后段局部 INT16 激活方案.精度由配置选择,不同方案的运行参数和结果分别记录.

## 评价协议

ESC-50 共 2000 条音频、20000 个窗口.第一折每类两条音频、每条前两窗构成 200 个校准窗口.主精度使用未参与校准的第 2-5 折,排除无直接标签映射的 `drinking_sipping`、`can_opening`、`washing_machine`,按 47 类、1504 条音频计算投影宏平均 AP,不等同于 AudioSet mAP.

前处理使用周期 Hann 400 点、帧移 160、FFT 512、64 Mel 频带、125–7500 Hz 和 `log(mel + 0.001)`.每个窗口 96 帧,步长 48 帧,尾段补零;每条音频对全部窗口的 521 类分数求均值.三个后端使用相同前处理和标签投影.

纯 NPU 延迟仅统计预热后的 `NeuronRuntime_inference`,不包含音频解码、特征提取、采集等待或指标汇总.RSS 为指定推理进程的峰值,不代表全部驱动内存.

## 结果

当前全 W8A16 交付指标见 [summary.json](results/summary.json):

| 指标 | 结果 |
| --- | ---: |
| ONNX FP32 投影 AP | 74.1456% |
| NPU 投影 AP | 73.8964% |
| 精度变化 | -0.2491 个百分点 |
| NPU 平均耗时 | 0.516 ms/窗口 |
| 推理进程峰值 RSS | 10.75 MiB |

报告分别记录交付运行和配对性能测量,不同测量条件的延迟不混用.局部精度方案与诊断数据保留在结果文件中,不在使用文档重复展示实验过程.

## 声音示例

| 输入 | 来源类别 | 输出 |
| --- | --- | --- |
| [sample_1.wav](examples/input/sample_1.wav) | dog | [JSON](examples/output/sample_1_events.json) |
| [sample_2.wav](examples/input/sample_2.wav) | glass_breaking | [JSON](examples/output/sample_2_events.json) |
| [sample_3.wav](examples/input/sample_3.wav) | siren | [JSON](examples/output/sample_3_events.json) |

仓库中的上述输出为 FP16 示例,不作为当前 W8A16 的评价结果.原始样例出处和许可见 [ESC-50 LICENSE](https://github.com/karolpiczak/ESC-50/blob/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/LICENSE).
