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

正式验证计划使用 LibriSpeech `test-clean` 报告英文 WER,使用 AISHELL-1 test 报告中文 CER.
评测前必须记录数据来源、许可证、清单哈希、文本规范化规则和运行 ID.校准数据与正式评测数据
必须分离.

批量评测代码和执行命令已经准备完成,但按用户分工尚未下载数据集或运行正式评测。实际结果
只能从 `summary.json` 中同步,不得根据脚本存在或静态检查推断 WER/CER。
