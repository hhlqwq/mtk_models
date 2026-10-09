# YAMNet / Genio 720

机器人环境声音分类,Google 原始 YAMNet,521 类 AudioSet 标签.固定网络输入
`1x1x96x64` Log-Mel 窗口,音频为 16 kHz 单声道.量化方式为 INT8 PTQ,
逐输出通道权重量化,CPU 音频前处理加 C++ Neuron Runtime NPU 推理.
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
200 窗 INT8 校准、`--arch=mdla5.3 --suppress-output --disallow-bridge`
DLA 编译、C++ 交叉编译和板端上传.不会自动下载或安装依赖.
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
200 窗校准;主指标只统计第二至第五折的 1504 条映射样本,同时报告全量指标.
该指标为 ESC-50 固定标签投影的宏平均 AP 与 Top-1,不等同于官方 AudioSet mAP.
AP 对同分数组统一处理,避免 INT8 并列分数导致乐观结果.

NPU 平均/P95 每窗计时只覆盖常驻模型的 `NeuronRuntime_inference`.
前处理与推理阶段分开计时,实时率使用两阶段实际处理时间之和,不宣称实时麦克风
端到端延迟.C++ 推理和 Python 前处理进程峰值 RSS 分别记录,不含 ffmpeg 子进程.

## 当前结果

正在实施,正式精度、性能和运行环境在板端全量完成后填入.

## 声音示例

编译时从 ESC-50 固定选择狗叫、玻璃破碎和警笛各一条,样例来源随部署上传.
板端输出原始 521 类 Top-5,取回后保存在 `examples/output/sample_*_events.json`.
示例音频许可归属见 ESC-50 原始 `LICENSE`,不将整个数据集纳入 Git.
