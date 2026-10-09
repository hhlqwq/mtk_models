# YAMNet / Genio 720

机器人环境声音分类,Google 原始 YAMNet,521 类 AudioSet 标签.固定网络输入
`1x1x96x64` Log-Mel 窗口,音频为 16 kHz 单声道.默认交付 `w8a16`: 权重 INT8,
全网络激活 INT16.同时保留 `int8_tail16`: 前段激活 INT8,第12层 pointwise 输出
至分类头使用 INT16 激活.两者均通过板端全量测试,详细对比见下节.
CPU 音频前处理加 C++ Neuron Runtime NPU 推理,FP16/W8A8 作为历史参考与对照.
高通 YAMNet 仅作为功能与交付参考,不使用高通权重或预编译模型.

## 保留的两种整数量化方案

模型总览只展示 **全 W8A16** 的交付指标,此处保留两个方案的详细结果.
两者使用相同Google模型、200窗第一折校准输入和ESC-50标签映射.
精度均来自2000条音频、20000窗口全量推理,主指标使用折2-5的1504条映射样本.

| 项目 | 全 W8A16,默认交付 | 第12层 pointwise 输出起 INT16,可选 |
| --- | ---: | ---: |
| 入口名称 | `w8a16` | `int8_tail16` |
| 权重 / 激活 | 全部INT8 / 全部INT16 | 全部INT8 / 前段INT8、尾部INT16 |
| 原生输入 / 输出 | INT16 / INT16 | INT8 / INT16 |
| 输入 / 输出有效字节数 | 12288 / 1042 | 6144 / 1042 |
| 主宏平均 AP | **73.8964%** | **73.4619%** |
| 主 Top-1 | 72.6064% | 72.8059% |
| AP 相对同协议 ONNX 变化 | -0.2491个百分点 | -0.6837个百分点 |
| 全量映射1880条 AP / Top-1 | 74.0222% / 73.0851% | 73.6609% / 73.2447% |
| 各自全量运行的NPU平均 / P95 | 0.516 / 0.620 ms | 0.493 / 0.597 ms |
| DLA大小 | 3986767 bytes | 4043567 bytes |
| 精度运行编号 | `20261009_yamnet_esc50_w8a16_v1` | `20261009_yamnet_tail16_diagnosis` |

ONNX参考主AP为74.1456%.局部INT16比全W8A16的AP低约0.4345个百分点,
其Top-1略高;AP与Top-1是不同指标,不据其中一个推断另一个.
上述两个全量运行发生在不同时间,耗时不可当作相同板端状态下的严格速度对比.
RSS、前处理耗时等完整交付数据见下方全W8A16结果;局部方案未单独归档的指标
不复用全W8A16的数据.

随后在92板端使用同一批前200条音频缓存,每轮2000窗,交替运行三轮,
每个模型合计6000窗,每次加载后先预热20次.计时仅覆盖硬件模式的
`NeuronRuntime_inference`调用,排除音频解码、特征提取和窗口等待.

| 同条件交替测试 | 全 W8A16 | 局部 INT16 |
| --- | ---: | ---: |
| NPU平均耗时 | 1.3195 ms | 1.2732 ms |
| 中位数 | 0.5615 ms | 0.5300 ms |
| P95 | 0.8012 ms | 0.7725 ms |
| P99 | 1.9780 ms | 2.1726 ms |
| 超过10 ms的窗口 | 33 / 6000 | 32 / 6000 |
| 最大单窗耗时 | 133.768 ms | 133.265 ms |

局部方案本次平均快0.0463 ms,约3.51%,中位数快0.0315 ms,约5.61%.
三轮局部平均分别为1.3613 / 1.2995 / 1.1588 ms,全W8A16为
1.3895 / 1.3178 / 1.2512 ms.两个模型都有少量明显慢于正常窗口的推理,
这些长尾将平均值拉高;具体原因未定位,不能据此认定某个算子或硬件缺陷.
本次观察支持局部方案略快,但速度收益较小,不作为固定性能保证.

全W8A16产物保留在板端 `/root/hailong.he/open_models/yamnet/models/`.
局部方案已验证产物保留在主机、Docker及板端
`/tmp/hailongcodex/2026-10-09/yamnet/operator_diagnosis/`,文件为
`tail16.tflite` / `tail16.dla` / `tail16_config.csv`;板端仅保留DLA和运行配置.
局部DLA SHA256为 `810b476ec61df9c83e8e099ecbe4eef1b8e01385525a5e34dfecec7772f929df`.
同条件交替测试的原始计时位于板端同目录的 `paired_w8a16_093230/`.
所有交付、局部精度和配对耗时汇总保留在唯一的 `results/summary.json`.

当前源码默认编译全W8A16,局部方案可用 `YAMNET_PRECISION=int8_tail16` 显式编译.
已有板端旧入口需同步新代码和局部产物后才能使用该精度名称;
上面的局部实测结果来自独立板端诊断程序,不是尚未执行的新入口测试.

## 来源和资源准备

官方源码: https://github.com/tensorflow/models/tree/34a21326906b9574fa11c4d6d0a5c534ff039267/research/audioset/yamnet

官方权重: https://storage.googleapis.com/audioset/yamnet.h5

源码和权重按原始 Google 实现使用,源码许可证 Apache-2.0.
具体下载地址、大小和 SHA256 见 [source_url.txt](models/source_url.txt).
ESC-50 原始资源保存 NAS,板端保留原始 WAV 与元数据.
ESC-50 整体为 CC BY-NC 3.0,各音频出处和许可见原始数据的 `LICENSE`.

在 89 仓库根目录准备资源:

```bash
python3 models/audio/ambient_sound/yamnet/deploy/host/prepare_assets.py \
  --models-dir /data/users/hailong.he/github/mtk_models/models/audio/ambient_sound/yamnet/models \
  --dataset-dir /data/users/hailong.he/nas_smb/Datasets/open_source/raw/ESC-50
```

若需要已有代理,使用 `--proxy socks5h://主机:端口`,脚本不修改代理配置.
下载和解压不会覆盖已有原始文件.各下载文件哈希记入本地资源清单.

导出依赖安装在临时目录的独立环境,不升级 Docker 全局包:

```bash
work=/tmp/hailongcodex/$(date +%F)/yamnet
docker exec hhl_g720_8011 python -m venv "$work/venv"
docker exec hhl_g720_8011 "$work/venv/bin/pip" install \
  tensorflow-cpu==2.15.1 tf2onnx==1.16.1 onnx==1.16.1 numpy==1.26.4 scipy==1.11.4 tqdm==4.66.5
docker exec hhl_g720_8011 "$work/venv/bin/pip" install --no-deps tf-keras==2.15.0
```

已有环境可用 `YAMNET_EXPORT_PYTHON` 指定实际 Python 路径.编译主机和板端需要
ffmpeg,板端 Python 需要 NumPy;MTK Converter 和 ONNX Runtime 复用现有 Docker.
本次隔离依赖的锁定版本和具体下载 URL、文件哈希分别保存在主机临时目录的
`requirements_locked.txt` 和 `dependency_sources.json`.

## 两步运行

Windows 修改提交后,89 拉取,在仓库根目录执行:

```bash
bash models/audio/ambient_sound/yamnet/deploy/run.sh
```

主机执行官方前向比较、固定图导出、TensorFlow 与 ONNX 全量精度评测,
默认全 W8A16 PTQ 转换、`--arch=mdla5.3 --suppress-output --disallow-bridge`
DLA 编译、C++ 交叉编译和板端上传.不会自动下载或安装依赖.
默认方案的原生输入和输出均为 INT16,Runtime 分别验证两者字节布局.
`YAMNET_PRECISION=fp16` 可显式使用原 FP16 流程.
可用 `YAMNET_PRECISION=int8 bash models/audio/ambient_sound/yamnet/deploy/run.sh`
执行 200 窗 PTQ 校准的 INT8 对照流程,当前 W8A8 配方存在明显退化,需继续优化.
W8A16 入口为 `YAMNET_PRECISION=w8a16 bash models/audio/ambient_sound/yamnet/deploy/run.sh`,
使用 INT8 权重、INT16 激活与原生 INT16 IO.转换后逐层检查 27 层卷积和 1 层
全连接的实际精度,板端全量结果见下方 W8A16 实测表.
各精度使用独立的 `runtime_config_精度.csv` 和 `quantization_精度.json`,
保留旧版 `runtime_config.csv` 作为既有部署的兼容入口.
主机评测与缓存写入 `/tmp/hailongcodex/当天日期/yamnet`,模型产物保存在 `models/`.

92 开发板执行:

```bash
EVAL_RUN_ID=自定义运行编号 bash /root/hailong.he/open_models/yamnet/run.sh
```

原始数据同步到 `/root/hailong.he/datasets/ESC-50/`.板端从 WAV 解码并生成
全部音频窗口,20 次预热后逐窗执行真实 NPU 推理,最后计算精度和示例结果.
运行目录 `/root/hailong.he/open_models/yamnet/results/运行编号/` 保留原始证据,
已有目录拒绝覆盖.仓库 `results/` 只保留最终 `summary.json`.

## 评测协议

采用官方周期 Hann 400 点、帧移 160 点、FFT 512 点、64 Mel 频带,
125-7500 Hz,`log(mel + 0.001)`,96 帧窗口、48 帧步长.尾段补零与官方一致.
三后端均经 ffmpeg 重采样,每条音频对全部窗口的 521 类分数求均值.

ESC-50 全部 2000 条均执行推理.标签投影固定在 `deploy/board/audio_utils.py`,
47 类具有直接对应标签;`drinking_sipping`、`can_opening`、`washing_machine`
没有直接对应标签,保留推理分数但不参与投影精度,不以近似标签冒充正确映射.
因此全量投影精度覆盖 1880 条、47 类.第一折每类两条音频、每条前两窗用于
200 窗整数校准;历史 FP16 不使用校准数据.所有精度的主指标都统计第二至第五折的
1504 条映射样本,同时报告全量指标.
该指标为 ESC-50 固定标签投影的宏平均 AP 与 Top-1,不等同于官方 AudioSet mAP.
AP 对同分数组统一处理,避免 INT8 并列分数导致乐观结果.

NPU 平均/P95 每窗计时只覆盖常驻模型的 `NeuronRuntime_inference`.
前处理与推理阶段分开计时,实时率使用两阶段实际处理时间之和,不宣称实时麦克风
端到端延迟.C++ 推理和 Python 前处理进程峰值 RSS 分别记录,不含 ffmpeg 子进程.

## 当前结果

正式运行编号: `20261009_yamnet_esc50_fp16_v2`,全部 **2000 条音频 / 20000 窗口**
真实 NPU 执行完成,逐音频哈希、标注、标签映射、窗口顺序和分数覆盖检查通过.
[summary.json](results/summary.json) 同时归档 FP16、W8A8 对照及 W8A16 全量结果.

| 后端 | 主评测宏平均 AP,1504 条 | 主评测 Top-1 | 全量投影 AP,1880 条 | 全量投影 Top-1 |
| --- | ---: | ---: | ---: | ---: |
| Google TensorFlow FP32 | 74.1457% | 73.7367% | 74.2705% | 74.1489% |
| ONNX Runtime FP32 | 74.1456% | 73.7367% | 74.2702% | 74.1489% |
| Genio 720 NPU FP16,历史参考 | 74.1440% | 73.8032% | 74.2760% | 74.2021% |
| Genio 720 NPU INT8,不推荐 | 43.9329% | 36.5691% | 43.5498% | 36.2234% |

FP16 相对同协议 ONNX 的主评测 AP 变化为 **-0.0016 个百分点**.
INT8 逐输出通道权重 PTQ 首轮运行 `20261009_yamnet_esc50_v1`,AP 下降
**30.2127 个百分点**,当前保留为量化对照而非默认部署.扩大校准窗口以及其他校准
实验未改善结果,当时采用 FP16,当前默认交付全 W8A16;FP16 为权重半精度压缩和硬件降精度执行,
无需 PTQ 校准,没有使用 Qualcomm 预导出模型或 CPU 推理桥接.

| FP16 板端性能项目 | 实测 |
| --- | ---: |
| NPU 平均 / P95 每窗推理 | 0.716 / 0.813 ms |
| 原始 WAV 解码及前处理平均,每条 5 秒音频 | 77.356 ms |
| 推理阶段平均,每条 10 窗,含缓存读取与 IO 登记 | 7.604 ms |
| 分阶段处理实时率 RTF,越低越快 | 0.01699 |
| C++ 推理进程峰值 RSS | 14.00 MiB |
| Python 前处理进程峰值 RSS | 40.89 MiB |
| DLA 大小 | 7638678 bytes,7.29 MiB |

音频窗口覆盖 0.96 秒 Mel 帧,含 STFT 尾部实际需要 15600 个采样点,每窗步长
0.48 秒;上表 NPU 耗时不含音频采集等待和前处理.RTF 为分阶段处理时间之和,
不是实时麦克风演示.两种进程 RSS 分别记录,不包含 ffmpeg 子进程.

一致性验证: 原始 Google 与固定窗口模型比较 10 段音频,最大分数差异
`2.74e-5`;TensorFlow 与 ONNX 全量最大分数差异 `2.22e-5`.NumPy 与官方
前处理最大特征差异 `1.35e-4`,主机与板端 10 段音频最大差异 `9.73e-5`.
100 ms 分块的流式缓存与离线窗口一致,尚未接入实际麦克风采集.

编译环境: 89 的 `hhl_g720_8011`,NeuroPilot SDK `8.0.11-build20260211`,
Converter `8.16.0`,NCC `8.2.31`,独立 TensorFlow CPU `2.15.1`,tf-keras `2.15.0`,
tf2onnx `1.16.1`,ONNX Runtime `1.18.0`.板端 Runtime 库 `8.2.16`,
Linux `6.6.137-mtk+ga246e0c68c39-g429091ed5965`,CPU governor `schedutil`,
ffmpeg `6.1.4`,NumPy `1.26.4`;未修改板端固件或全局依赖.
原生 FP16 输入为 **12288 bytes**,输出为 **1042 bytes**,CPU 仅前处理及结果汇总.

产物 SHA256:

| 产物 | bytes | SHA256 |
| --- | ---: | --- |
| ONNX FP32 | 14935577 | `a2deca938ea590b8fe2643408ea9df73a956e482b0a5525308f930f71ca301a8` |
| FP16 TFLite | 7498828 | `8fb178803a3fc6d79646f8174b2ea64ea4d182979c92aebf5db818d5915f701f` |
| FP16 DLA | 7638678 | `5c5894c1ca938730961da0ce6f0ecc21221c67aef4840a3d33e0600333d0c69d` |

板端详细证据留在 `/root/hailong.he/open_models/yamnet/results/20261009_yamnet_esc50_fp16_v2/`,包含实际分数、逐窗耗时、
音频哈希与日志;Git 中 `results/` 仅保留 `summary.json`.主机浮点结果与隔离依赖
来源记录留在 `/tmp/hailongcodex/2026-10-09/yamnet/`,浮点结果位于 Docker 同名临时
目录,依赖锁定与下载来源记录位于 89 主机临时目录.

## W8A16 板端全量结果

运行 `20261009_yamnet_esc50_w8a16_v1`,原始 WAV 前处理和全部 **2000 条音频 /
20000 窗口**硬件推理完成.与原有后端采用相同原始音频、标签映射和评测协议,
第一折相同 100 条音频的前两窗用于 200 窗校准,主精度仍使用折 2-5 的 1504 条样本.

| 项目 | W8A16 实测 |
| --- | ---: |
| 主评测宏平均 AP / Top-1 | 73.8964% / 72.6064% |
| 全量投影 AP / Top-1,1880 条 | 74.0222% / 73.0851% |
| AP 相对 ONNX 变化 | -0.2491 个百分点 |
| NPU 平均 / P95 每窗推理 | 0.516 / 0.620 ms |
| 每段音频前处理 / 推理阶段平均 | 80.938 / 6.096 ms |
| 分阶段处理 RTF | 0.01741 |
| C++ 推理 / Python 前处理峰值 RSS | 10.75 / 40.12 MiB |
| DLA 大小 | 3986767 bytes,3.80 MiB |

转换后核验 **28 个主计算层**的 INT8 权重、INT16 激活与 INT32 偏置;
全部张量为 INT8 28 个、INT16 33 个、INT32 31 个,无浮点张量.
NCC 使用 `--arch=mdla5.3 --suppress-output --disallow-bridge`,Runtime 使用硬件模式.
原生输入为 `12288 bytes`,输出为 `1042 bytes`;两者均为有符号 INT16,
不是 FP16.输入 scale/zero_point 为 `0.00017911930626723915 / 5797`,
输出为 `0.0000152587890625 / -32768`.

W8A16 TFLite SHA256: `03751967e37fb901b27f302465dff279dc85f96fde970822989dcc0fb1366b42`.

W8A16 DLA SHA256: `04692e6ec7cec50b8680aa36231699706f41ebd3948497ddc2d75b2f31238caf`.

原始分数、逐窗计时、三条声音示例和哈希证据位于板端
`/root/hailong.he/open_models/yamnet/results/20261009_yamnet_esc50_w8a16_v1/`.
W8A16 示例 Top-5 已内嵌唯一结果文件的 `examples`,原有示例文件保留 FP16 输出.
新 Runtime 对原 FP16 和 W8A8 分别全量回归,每种均为 2000 条 / 20000 窗,
全部分数与原结果逐值一致.当前默认入口为全 W8A16,局部 INT16 作为可选方案保留.
已上传两种产物后,板端可显式切换,不用重新编译:

```bash
YAMNET_PRECISION=w8a16 EVAL_RUN_ID=新的运行编号 \
  bash /root/hailong.he/open_models/yamnet/run.sh
```

### 与高通量化参考的关系

[高通官方 YamNet 模型卡](https://huggingface.co/qualcomm/YamNet) 同时列出 W8A8
和 W8A16,两者权重均为 INT8,激活位宽分别为 8 和 16.其 QCS6490 性能表中的
TFLite W8A8 为 0.505 ms,ONNX W8A16 为 0.722 ms,对应用户参考截图的两条路径.
这些是该设备的部署性能,没有提供本项目同协议 ESC-50 投影 AP.
[官方源码封装](https://github.com/qualcomm/ai-hub-models/blob/v0.63.0/src/qai_hub_models/models/yamnet/model.py)
采用 `torch_audioset` 版本,校准数据接口为 AudioSet;本项目继续采用 Google 官方
TensorFlow 源码与原始权重,没有使用高通预导出产物作为移植源.

当前 MTK W8A8 配方的 AP 43.9329% 只能说明该配方尚未达到精度目标,
不能推断 YAMNet 无法做 INT8 部署.W8A16 使用 INT8 权重已取得 AP 73.8964%,
支持优先排查 W8A8 激活量化尺度和逐层误差的判断;本次已完成以下定位与对照验证.

## W8A8 精度下降定位与算子改法验证

2026-10-09 在原 Google 模型、原 200 窗校准集和相同标签映射上完成诊断.
主因是 **第 12 层 pointwise 输出与第 13 层 depthwise 输出的小激活被 INT8
量化舍入为零**,随后在末端卷积放大;单独替换 Sigmoid 或平均池化未解决问题.

小批诊断固定使用第 2 折每类首条音频,共 50 条 / 500 窗,其中 47 类有直接映射.
以下为主机模拟和因果对照,不能代替板端全量指标:

| 小批对照 | 投影宏平均 AP |
| --- | ---: |
| ONNX FP32 | 79.7163% |
| 原 W8A8 模拟 | 46.4882% |
| 仅将最终 FP32 概率量化为 INT8 | 78.6018% |
| 仅量化 FP32 logits,再计算 Sigmoid 和输出量化 | 78.7098% |
| INT8 主干与 logits,CPU 浮点 Sigmoid | 49.9184% |
| INT8 主干,浮点全连接与 Sigmoid | 47.4691% |
| INT8 主干,浮点池化、全连接与 Sigmoid | 47.4530% |
| 平均池化替换为等价 depthwise Conv,纯 W8A8 | 46.6872% |
| Sigmoid 前增加 logits 范围限制 [-12,12],纯 W8A8 | 46.4882% |

逐层相对 RMSE 在第 12 层 pointwise 约 10.17%,第 13 层 pointwise 约 41.17%,
第 14 层 depthwise 约 170.78%.误差跳变的位置与信息最先丢失的位置并不相同:
对原始浮点激活 **只插入一次 INT8 量化,前后均用浮点计算**,第 12 层 pointwise
输出量化后的小批 AP 降至 59.4512%,第 13 层 depthwise 输出量化后降至 45.3931%.
两个边界分别有 72.8739% / 66.3414% 的正激活被舍入为零,饱和比例仅约
0.00417% / 0.00098%.这些比例针对本次固定小批,说明主要问题是小值分辨率,
而不是大量超出量化范围的截断.

分类 logits 范围约 [-600.95,24.53],INT8 步长约 2.45,确实很粗;
但仅对浮点 logits 做同尺度量化的对照没有复现主干的严重下降.
额外 Clip 的输出步长虽减小到约 0.094,上游全连接输出仍先使用 2.45 的步长,
已丢失的信息没有恢复.池化替换的浮点最大分数误差为 `1.79e-7`,Clip 为
`6.11e-6`,两者均通过浮点功能检查,量化后仍未恢复精度.

对敏感位置继续进行局部精度和有界通道重参数化验证.重参数化仅作用于上述
两个边界,缩放激活对应的前层权重/偏置并反向缩放后层权重,排除极小或无响应
通道,因子范围 [1,16];浮点最大分数误差 `5.36e-7`,量化仍为纯 W8A8.

| 板端全量方案,2000 条 / 20000 窗 | 主 AP,折 2-5 | 主 Top-1 |
| --- | ---: | ---: |
| ONNX FP32 参考 | 74.1456% | 73.7367% |
| 原 W8A8 | 43.9329% | 36.5691% |
| 两个敏感边界做等价通道重参数化,纯 W8A8 | 66.4728% | 66.2899% |
| 仅两个敏感边界使用 INT16 激活,其余 INT8 | 71.9474% | 70.2793% |
| 第 12 层 pointwise 输出至分类头使用 INT16 激活 | 73.4619% | 72.8059% |
| 全模型 W8A16,原已交付方案 | 73.8964% | 72.6064% |

最小候选实际只有 `layer12/pointwise_conv/relu/Relu:0_ndsc_0` 和
`layer13/depthwise_conv/relu/Relu:0_ndsc_0` 两个 INT16 激活输出,28 个主计算层
的权重仍为 INT8,原生输入/输出均为 INT8.仅提高第 14 块两个输入边界的小批
AP 为 47.1415%,没有恢复精度,进一步支持第 12/13 块敏感边界的定位.
宽范围候选的输入为 INT8、输出为 INT16.两种局部方案均为混合激活位宽,
不能标为纯 W8A8.全部候选采用硬件 Runtime,编译包含 `--disallow-bridge`.

**可行结论:** 等价通道重参数化已将纯 W8A8 全量 AP 提升约 22.54 个百分点,
但仍落后参考约 7.67 个百分点.局部 INT16 可明显恢复精度,其中宽范围方案与
参考相差约 0.68 个百分点;现有全 W8A16 的差距更小.不应直接删除主干卷积或
ReLU 来修复量化误差,这样会改变原模型函数.当前已验证结果优先支持保留模型
结构并调整敏感激活精度,没有证据证明只移除一个算子就能恢复纯 W8A8 精度.

本次最早宽范围候选全量平均为 0.493 ms / 窗,后续候选及短时复测共同出现
约 9 ms 的 P95 长尾,三个候选复测平均均约 3 ms.因此当前不据此宣称候选
更快,需固定板端运行状态后再比较性能.精度结果、完整窗口覆盖和真实张量
类型均已核验;所有原始计时保留,不将较小的一次耗时替换为统一性能结论.

用户选择保留全W8A16和宽范围局部INT16两种方案,默认交付为 `w8a16`,
可选入口名称为 `int8_tail16`.局部转换入口已接入源码,新入口的正式同步与回归
尚未执行;此节的精度与计时来自已经完成的板端诊断运行.
诊断汇总内嵌 [results/summary.json](results/summary.json) 的 `int8_operator_diagnosis`,
`results/` 仍只含这一个文件.临时脚本、候选 ONNX/TFLite/DLA、校准配置、原始
分数与日志位于 89 主机及 Docker 的
`/tmp/hailongcodex/2026-10-09/yamnet/operator_diagnosis/`;
板端同名目录保存三个候选的 `*_full_work` 和短时性能对照.
全量工作目录的特征缓存为原 FP16 已验证缓存的只读符号链接,没有修改原始音频.

## 声音示例

编译时从 ESC-50 固定选择狗叫、玻璃破碎和警笛各一条,样例来源随部署上传.
板端输出原始 521 类 Top-5,取回后保存在 `examples/output/sample_*_events.json`.
示例音频许可归属见 ESC-50 原始 `LICENSE`,不将整个数据集纳入 Git.

| 输入与原始 ESC-50 文件 | 类别 | 原始 521 类 Top-5 分数,FP16 板端实际输出 | 输出 |
| --- | --- | --- | --- |
| [sample_1.wav](examples/input/sample_1.wav),`1-100032-A-0.wav` | dog | Silence: 0.7020; Dog: 0.1093; Domestic animals, pets: 0.1067; Animal: 0.0849; Bow-wow: 0.0805 | [JSON](examples/output/sample_1_events.json) |
| [sample_2.wav](examples/input/sample_2.wav),`1-20133-A-39.wav` | glass_breaking | Silence: 0.3021; Coin (dropping): 0.1169; Glass: 0.1055; Cutlery, silverware: 0.0752; Dishes, pots, and pans: 0.0736 | [JSON](examples/output/sample_2_events.json) |
| [sample_3.wav](examples/input/sample_3.wav),`1-31482-A-42.wav` | siren | Siren: 0.2582; Opera: 0.2390; Alarm: 0.2340; Classical music: 0.1889; Emergency vehicle: 0.1812 | [JSON](examples/output/sample_3_events.json) |

示例对 5 秒音频的 10 窗分数求均值,短促声音可能被静音背景主导;以上保留真实输出,
没有按期望类别修改模型分数.样例逐条归属和许可见 [ESC-50 原始 LICENSE](https://github.com/karolpiczak/ESC-50/blob/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/LICENSE).
