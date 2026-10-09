# YAMNet / Genio 720

机器人环境声音分类,Google 原始 YAMNet,521 类 AudioSet 标签.固定网络输入
`1x1x96x64` Log-Mel 窗口,音频为 16 kHz 单声道.默认 FP16 权重压缩与 NPU
降精度编译,保留可选 INT8 PTQ,CPU 音频前处理加 C++ Neuron Runtime NPU 推理.
高通 YAMNet 仅作为功能与交付参考,不使用高通权重或预编译模型.

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
默认 FP16 转换、`--arch=mdla5.3 --relax-fp32 --suppress-input --suppress-output
--disallow-bridge` DLA 编译、C++ 交叉编译和板端上传.不会自动下载或安装依赖.
可用 `YAMNET_PRECISION=int8 bash models/audio/ambient_sound/yamnet/deploy/run.sh`
执行 200 窗 PTQ 校准的 INT8 对照流程,该精度存在明显退化,当前不推荐部署.
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
200 窗 INT8 校准;默认 FP16 不使用校准数据.两种精度的主指标都统计第二至第五折的
1504 条映射样本,同时报告全量指标.
该指标为 ESC-50 固定标签投影的宏平均 AP 与 Top-1,不等同于官方 AudioSet mAP.
AP 对同分数组统一处理,避免 INT8 并列分数导致乐观结果.

NPU 平均/P95 每窗计时只覆盖常驻模型的 `NeuronRuntime_inference`.
前处理与推理阶段分开计时,实时率使用两阶段实际处理时间之和,不宣称实时麦克风
端到端延迟.C++ 推理和 Python 前处理进程峰值 RSS 分别记录,不含 ffmpeg 子进程.

## 当前结果

正式运行编号: `20261009_yamnet_esc50_fp16_v2`,全部 **2000 条音频 / 20000 窗口**
真实 NPU 执行完成,逐音频哈希、标注、标签映射、窗口顺序和分数覆盖检查通过.
[summary.json](results/summary.json) 同时归档 FP16 正式结果与 INT8 对照结果.

| 后端 | 主评测宏平均 AP,1504 条 | 主评测 Top-1 | 全量投影 AP,1880 条 | 全量投影 Top-1 |
| --- | ---: | ---: | ---: | ---: |
| Google TensorFlow FP32 | 74.1457% | 73.7367% | 74.2705% | 74.1489% |
| ONNX Runtime FP32 | 74.1456% | 73.7367% | 74.2702% | 74.1489% |
| Genio 720 NPU FP16,默认部署 | 74.1440% | 73.8032% | 74.2760% | 74.2021% |
| Genio 720 NPU INT8,不推荐 | 43.9329% | 36.5691% | 43.5498% | 36.2234% |

FP16 相对同协议 ONNX 的主评测 AP 变化为 **-0.0016 个百分点**.
INT8 逐输出通道权重 PTQ 首轮运行 `20261009_yamnet_esc50_v1`,AP 下降
**30.2127 个百分点**,当前保留为量化对照而非默认部署.扩大校准窗口以及其他校准
实验未改善结果,正式交付选用 FP16;FP16 为权重半精度压缩和硬件降精度执行,
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
