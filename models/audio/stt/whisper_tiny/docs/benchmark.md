# 性能验证

## 2026-09-16 单次 Smoke Test

| Run ID | 输入 | Encoder | Decoder 调用 | Decoder 总计 | Decoder 均值 | 生成 Token |
|---|---|---:|---:|---:|---:|---:|
| `20260916_jfk_fp32_v1` | OpenAI JFK | 81.6923 ms | 27 | 277.451 ms | 10.2759 ms | 23 |
| `20260916_silence_fp32_v1` | 5 秒纯静音 | 81.7182 ms | 5 | 48.6433 ms | 9.72866 ms | 1 |

环境为 Genio 720 / MT8189、Linux 6.6.117、Neuron Runtime 8.2.16；模型由 NCC 8.2.31
以 `mdla5.3`、`--suppress-input --suppress-output --disallow-bridge` 编译.计时只覆盖
`NeuronRuntime_inference`,不包含 DLA 加载、音频解码、Log-Mel 和 Token 文本解码.
Decoder 调用数包含 4 个初始提示 Token,因此高于生成 Token 数.

这些是两次单样例 Smoke Test,用于证明真实 NPU 可执行和量级,不是正式 Benchmark.

正式报告必须包含音频读取/重采样、Log-Mel、Encoder、首 Token、Decoder 单 Token、端到端
耗时、RTF、Tokens/s、加载时间和峰值 RSS,并按 1、5、15、30 秒音频分组报告
Mean/P50/P90/P95.
