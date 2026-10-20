# Whisper-Tiny

多语言语音识别。输入为 16 kHz 单声道音频，板端运行 Encoder 和 Decoder 两个 DLA。模型来源见[模型卡](model_card.md)。

量化方式: 未进行整数定点量化,采用浮点模型部署.

## 第一步: 编译并上传

模型、数据集和部署路径见 [config.yaml](config.yaml),工具链与 NAS 根目录按[仓库说明](../../../../README.md)配置.

在编译主机的本模型目录运行:

```bash
bash deploy/run.sh
```

转换、双 DLA 编译、参考评价和上传均在当前容器内完成。

双 DLA 编译限定 `mdla5.3`,禁止桥接并保留浮点输入输出.出现不支持的算子时编译直接失败,不上传含板端不支持目标的模型.旧 DLA 若报 `EDPA_1_2` 或加载失败,需要重新编译两个模型后上传.

## 第二步: 开发板测试

登录所配置的开发板，执行脚本打印的 `bash .../run.sh` 命令。板端准备 2620 条音频、运行 C++ 推理并计算 WER；结果写入 `BOARD_RESULTS_DIR`。

进入实际配置的部署目录后也可运行:

```bash
bash run.sh
```

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

全量测试自动复用选定样本的板端预测,生成少量效果文件到本次结果目录的 `examples/output/`.
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
├── run.sh
├── board_paths.conf
├── models/              # 模型与推理所需参数.
├── board/               # 板端程序、评测代码及必要依赖.
├── examples/input/      # 少量示例输入及来源清单.
└── results/             # 汇总及少量效果示例.
```

全量测试结果保存在 `BOARD_RESULTS_DIR/<运行编号>/summary.json`.成功后保留汇总和少量效果示例,失败时保留本次工作目录.将汇总上传为本模型的 `results/summary.json` 后更新 README.
## 文件结构

- `deploy/run.sh`: 编译上传和板端测试的唯一 Shell 入口.
- `deploy/host/`: 编译主机使用的导出、转换与辅助工具.
- `deploy/board/`: 板端程序源码、预处理和评测代码.
- `models/`: 原始权重、转换产物与来源说明; 必要的上游源码放在 `models/upstream/`.
- `examples/`: 少量固定输入与实际板端效果输出.
- `results/summary.json`: 上传后的最新测试汇总.
