# Whisper-Tiny

本目录用于将 OpenAI 多语言 Whisper-Tiny 部署到 MT8189 / Genio 720.当前状态为
`板端已验证`：双 DLA 已在 Neuron Runtime 8.2.16 完成英文与静音 Smoke Test,
Token 和文本均与 OpenAI FP32 Greedy Search 基线完全一致.正式 WER/CER 与完整性能评测待补.

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
NCC_MODE=strict bash models/audio/stt/whisper_tiny/deploy/build.sh
```

严格编译同时使用 `--suppress-input --suppress-output --disallow-bridge`，避免编译器插入
外部布局转换桥接.板端 Runtime 探针确认原生 I/O 为 FP16；程序按 Runtime 返回的 padded
字节数和布局填充缓冲区,不能把 TFLite 的 FP32 字节数直接传给 DLA.
`deploy/build_board_cpp.sh` 会交叉编译板端 I/O 探针，用于在真实 Runtime 上核对每个
输入输出的硬件对齐字节数和四维布局，并生成 `whisper_board_decode` 双 DLA 解码程序。
`deploy/prepare_board_inputs.py` 在 89 上生成 FP16 Mel、OpenAI FP32 基线和固定解码规则；
`deploy/decode_board_tokens.py` 将板端 Token 解码为文本并执行精确对比。
2026-09-16 的板端证据见 `docs/board_smoke_20260916.json`、`docs/accuracy.md` 和
`docs/benchmark.md`.完整交付还要求 LibriSpeech WER、AISHELL-1 CER、分组延迟/RTF、
峰值 RSS 和中文/噪声样例.

## 验证边界

- 已验证：OpenAI FP32/改写图/ONNX 数值对齐、MTK Converter 8.16.0 转换、NCC 8.2.31
  双图单一 MDLA 5.3 执行步、禁止 bridge 编译、Genio 720 双 DLA 完整解码.
- 已验证样例：OpenAI `jfk.flac` 的 23 个 Token 和文本逐项一致；生成的 5 秒静音样例
  输出 1 个 Token `291`（文本 `you`）,与 OpenAI FP32 基线一致且不同于 JFK 输出.
- 未验证：LibriSpeech WER、AISHELL-1 CER、中文/噪声样例、正式重复性能统计、RTF 和峰值 RSS.
