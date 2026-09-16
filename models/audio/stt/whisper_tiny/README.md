# Whisper-Tiny

本目录用于将 OpenAI 多语言 Whisper-Tiny 部署到 MT8189 / Genio 720.当前状态为
`环境建设中`：来源、目录、离线导出和转换入口已经建立,但尚未生成 ONNX/TFLite/DLA,
也尚未取得板端推理、WER/CER 或性能结果.

## 首版范围

- 输入：16 kHz 单声道 PCM/WAV,单段最长 30 秒.
- 输出：中文或英文转写文本,使用 Greedy Search.
- 图拆分：`encoder_fp32.onnx` 与 `decoder_step_fp32.onnx`.
- Decoder：单 Token 自回归,固定 200 Token KV Cache；每层 Key/Value 使用相同的
  `[H, 1, D, M]` rank-4 I/O 布局.
- Token Embedding 使用主机 One-Hot + MatMul,规避 MDLA 不支持的 `GATHER`.
- 导出时关闭 Torch SDPA,使用 OpenAI 官方 MatMul/Softmax 注意力分支,兼容 Torch 2.0 ONNX.
- 首版不包含流式麦克风、长音频滑窗、时间戳、翻译和说话人分离.

## 来源边界

正式权重仅使用 OpenAI `tiny.pt`,版本和 SHA-256 见 [model.yaml](model.yaml).Qualcomm
Whisper-Tiny 只用于参考 Encoder/Decoder 拆分与交付展示,不使用其 ONNX、QNN 或其他预转换产物.

## 离线准备

项目规定权重由用户下载.请将 `tiny.pt` 放到 `original/tiny.pt`,然后在 Ubuntu89 的
`hhl_g720_8011` 容器内执行：

```bash
cd /data/users/hailong.he/github/mtk_models
source env.sh
bash models/audio/stt/whisper_tiny/deploy/download_original.sh
```

脚本依次校验官方 SHA-256、导出 Encoder 和 Decoder-Step、检查 ONNX 图,并生成
`models/SHA256SUMS`.它不包含任何网络下载.

数值对齐分别约束 Encoder 的最大/均值绝对误差与余弦相似度,以及 Decoder logits 和
KV Cache 的逐 Token 最大绝对误差,避免用单一宽松 `allclose` 掩盖局部偏差.

导出环境依赖见 `deploy/requirements-export.txt`.2026-09-15 已从 Windows 下载固定依赖,
复制到 89 并在 `hhl_g720_8011` 中离线安装；`whisper --help` 和 `pip check` 已通过.
固定依赖文件的精确下载地址与 SHA-256 记录在 `deploy/offline_dependencies.lock`,89 与
92 均未直接联网安装.环境验证证据见 `docs/environment_setup_20260915.json`.

## 转换和编译

真实图审计通过后执行：

```bash
bash models/audio/stt/whisper_tiny/deploy/convert.sh
bash models/audio/stt/whisper_tiny/deploy/build.sh
```

首次运行应保留 Converter/NCC 完整日志并检查执行计划.只有 Encoder 与 Decoder 循环都在
92 板端生成正确文本,才可升级为"板端已验证".完整交付还要求 LibriSpeech WER、
AISHELL-1 CER、RTF、Token 延迟、峰值 RSS、文件哈希和运行 ID.

## 当前未验证项

- OpenAI FP32 与 ONNX 的 Mel、Encoder、逐 Token logits、Token 序列和文本一致性.
- MTK Converter 8.16.0 对 Decoder 固定 KV Cache 图的兼容性.
- NCC 8.2.31 的 MDLA 5.3 全图执行计划.
- Genio 720 Neuron Runtime 8.2.16 完整解码和性能.
- 正式中英文 WER/CER.
