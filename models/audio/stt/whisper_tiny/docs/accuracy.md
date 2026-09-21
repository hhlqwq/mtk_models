# 精度验证

## 2026-09-16 数值与板端 Smoke Test

- Encoder OpenAI FP32/ONNX：最大绝对误差 `0.0031523705`,均值绝对误差
  `4.6973692e-06`,P99 `4.1007996e-05`,余弦相似度 `0.9999999999`.
- Decoder 四步 OpenAI/改写图最大 Logits 误差 `3.9100647e-05`；改写图/ONNX 最大
  Logits 误差 `3.2901764e-05`；最大 Cache 误差 `6.5565109e-06`.
- Run `20260916_jfk_fp32_v1`：Genio 720 输出 23 个 Token,与 OpenAI FP32 基线
  逐项完全一致；文本完全一致：
  `And so my fellow Americans ask not what your country can do for you ask what you can do for your country.`
- Run `20260916_silence_fp32_v1`：本地生成 5 秒纯静音,板端与 OpenAI FP32 均输出
  Token `291`（文本 `you`）.该结果用于排除 JFK 输出残留,不代表具备静音拒识能力.

## 2026-09-18 AISHELL-1 test 正式评测

| 项目 | NPU | OpenAI CUDA 基线 |
|---|---:|---:|
| 样例 | 7,176 / 7,176 | 7,176 / 7,176 |
| CER | 45.593471% | 45.826373% |
| 错误数 / 参考字符 | 47,766 / 104,765 | 48,010 / 104,765 |
| 替换 / 删除 / 插入 | 44,442 / 990 / 2,334 | 44,437 / 993 / 2,580 |
| 规范化文本完全正确 | 198（2.759197%） | 201（2.801003%） |

- Run ID：`20260918_aishell1_test_fp16_v1`.
- 执行完整性：状态 `complete`,失败 0,缺失 0,超过 30 秒排除 0.
- 协议：OpenAI `BasicTextNormalizer` 后移除空白,不做繁简转换,按 Unicode 字符计算 CER；
  中文 `transcribe`、无时间戳、Greedy Search、最多 196 个生成 Token.
- NPU/OpenAI 一致性：规范化文本完全一致 `6,764/7,176`（94.258640%）；Token 完全一致
  `6,751/7,176`（94.077480%）.
- NPU CER 比同协议 OpenAI 基线低 0.232902 个百分点,但存在 412 条文本不完全一致,不能将
  该聚合差值解释为 NPU 精度提升.
- 14 条 NPU 样例未生成 EOT,均出现重复输出并达到生成 Token 上限；这些输出未剔除,已计入
  上述 CER 和性能统计.
- 数据：OpenSLR SLR33 AISHELL-1 test,Apache License 2.0；原始压缩包 SHA-256
  `a4a0313cde0a933e0e01a451f77de0a23d6c942f4694af5bb7f40b9dc38143fe`.
- 清单 SHA-256：`ac8396326581765a4afda4a333bb297b2ec1aba387e72d029ce8b10030d32ab6`.

结论：板端正式 Run 已完整执行,整体 CER 与同协议 OpenAI Tiny 基线接近,但绝对 CER 较高、
大规模 Token 完全一致率未达到 100%,且存在 14 条重复解码。当前结果用于如实记录
Whisper-Tiny FP16/Greedy Search 的部署表现,不标记为精度验收通过。LibriSpeech
`test-clean` WER 仍未运行.

完整机器可读证据见 [`formal_eval_20260918_aishell1.json`](formal_eval_20260918_aishell1.json).
