# Whisper-Tiny

> Python 推理代码已迁移,尚未做板端验证.下文报告和示例属于迁移前的 C++/原生 CLI 实现,不能作为当前 Python 实现的精度、性能或 RSS.

多语言语音识别。输入为 16 kHz 单声道音频，板端运行 Encoder 和 Decoder 两个 DLA。模型来源见[模型卡](model_card.md)。

量化方式: 未进行整数定点量化,采用浮点模型部署.

## 执行范围

先在已有编译容器执行 `bash deploy/compile.sh`,完成转换、量化、编译与上传后结束.随后登录开发板,在部署目录执行 `bash compile.sh`,做少量通路检查并保存 README 示例到 `examples/output/`.该阶段不计算或发布精度、性能和峰值 RSS 报告.

正式评价使用独立的 `evaluate.sh`: 容器端只计算已有浮点模型的同协议参考及必要评价输入,资源单独放入已有部署包后,板端运行 `bash evaluate.sh`,统计精度、性能和推理进程峰值 RSS.正式结果只写入 `results/summary.json`,不重新量化或编译.预测、日志及内部指标位于当天日期临时目录.详见[板端规范](../../../../docs/genio_720.md).

下文已归档结果保留原始模型版本和评价范围,本次入口调整不代表重新实测.

## 第一步: 编译并上传

模型、数据集和部署路径见 [config.yaml](config.yaml),工具链与 NAS 根目录按[仓库说明](../../../../README.md)配置.

在编译主机的本模型目录运行:

```bash
bash deploy/compile.sh
```

转换、双 DLA 编译、参考评价和上传均在当前容器内完成。

双 DLA 编译限定 `mdla5.3`,禁止桥接并保留浮点输入输出.出现不支持的算子时编译直接失败,不上传含板端不支持目标的模型.旧 DLA 若报 `EDPA_1_2` 或加载失败,需要重新编译两个模型后上传.

## 第二步: 独立评价

上传完成后,请登录开发板,在部署目录执行 `bash compile.sh` 完成通路检查和示例生成.正式评价前,按[板端规范](../../../../docs/genio_720.md)准备同协议参考及必要输入,在已部署目录执行:

```bash
bash evaluate.sh
```

精度、性能和峰值 RSS 统一保存到 results/summary.json,过程数据位于日期临时目录,不重新量化或编译.

## 数据与精度评测

参考评价使用 `config.yaml` 指定的数据集,与板端保持相同样本、标注和预后处理协议.

数据目录要求: LibriSpeech test-clean: 保留官方目录下的 FLAC 音频和 *.trans.txt,2620 条音频.

精度变化以百分点表示,正数为改善,负数为下降.

Docker 需要 ONNX Runtime、NumPy、OpenCV、tqdm、ffmpeg 和 OpenAI Whisper。Whisper 仅提供音频处理、分词及文本规范化,模型推理由 ONNX Runtime 执行。
仓库 Docker 镜像包含 ffmpeg; 现有容器若缺少该依赖,需在容器中执行 `apt-get update && apt-get install -y --no-install-recommends ffmpeg`.脚本会在模型转换前检查该依赖.

## 当前测试结果

LibriSpeech test-clean 全量 2620 条音频,结果见 [summary.json](results/summary.json)。

| 指标 | 结果 |
| --- | ---: |
| ONNX FP32 WER | **7.56%** |
| 板端 WER | **7.56%** |
| 精度变化 | **下降不足 0.01 个百分点** |
| 板端 NPU 平均耗时 | **379.19 ms/条音频** |
| 推理进程峰值 RSS | **105.53 MiB** |

WER 越低越好,本次板端 WER 相比 ONNX 上升不足 0.01 个百分点。精度变化按 ONNX WER 减板端 WER 计算,完整数值保存在汇总文件中。

耗时为每条音频的 Encoder 和全部自回归 Decoder NPU 调用之和,不包含前后处理,不等同于一次 Decoder 调用耗时。峰值 RSS 为双模型常驻推理进程的峰值,包含运行库及前后处理,不代表 NPU 专用内存。

## 效果示例

提供两位说话人的短音频,输入选自 [LibriSpeech test-clean](https://www.openslr.org/12/),对应输出来自本次板端全量测试。

输入已放在 `examples/input/`,编译时从配置的数据集自动生成来源与样本清单,随部署上传,不纳入 Git.

compile.sh 复用固定少量样本的板端预测,生成展示文件到部署目录的 `examples/output/`,不依赖全量评价.
输出目录保留一个 [transcripts.md](examples/output/transcripts.md),集中展示两条音频的实际板端识别结果。

```text
examples/
├── input/sample_1.wav
├── input/sample_2.wav
└── output/transcripts.md
```

### 示例 1

[播放或下载音频](examples/input/sample_1.wav)

参考文本: HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE

板端识别: He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fat and sauce.

### 示例 2

[播放或下载音频](examples/input/sample_2.wav)

参考文本: YOU WILL FIND ME CONTINUALLY SPEAKING OF FOUR MEN TITIAN HOLBEIN TURNER AND TINTORET IN ALMOST THE SAME TERMS

板端识别: You will find me continually speaking of foreman, Titian, Holbein, Turner, and Tintarat, and almost the same terms.

输出文件: [examples/output/transcripts.md](examples/output/transcripts.md),包含上述两条音频的参考文本与实际板端识别文本。

## 板端部署结构

第一步上传到 `BOARD_DEPLOY_DIR` 后的布局统一为:

```text
部署目录/
├── compile.sh
├── evaluate.sh          # 独立精度、性能与 RSS 评价.
├── config.json          # 不含连接地址的模型配置.
├── models/              # 模型与推理所需参数.
├── board/               # 板端程序、评测代码及必要依赖.
├── examples/            # 少量输入、来源清单和展示输出.
└── results/summary.json  # 唯一正式评价报告.
```

评价结果只保存在部署目录的 `results/summary.json`.预测、日志和中间指标保留在 `/tmp/hailongcodex/<当天日期>/<模型>_evaluate_<运行编号>/`;展示示例保存在 `examples/output/`.取回唯一报告后据实更新 README.
## 文件结构

- `deploy/compile.sh`: 编译部署与展示示例入口;精度、性能和 RSS 使用独立的 `deploy/evaluate.sh`.
- `deploy/host/`: 编译主机使用的导出、转换与辅助工具.
- `deploy/board/`: 板端程序源码、预处理和评测代码.
- `models/`: 原始权重、转换产物与来源说明; 必要的上游源码放在 `models/upstream/`.
- `examples/`: 少量固定输入与实际板端效果输出.
- `results/summary.json`: 上传后的最新测试汇总.
