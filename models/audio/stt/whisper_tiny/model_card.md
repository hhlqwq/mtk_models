# Model Card：Whisper-Tiny

## 模型

- 上游：`openai/whisper`.
- 版本：`v20250625`.
- 权重：多语言 `tiny.pt`,约 39M 参数.
- 许可证：MIT.
- 目标：MT8189 / Genio 720.

## 预处理与解码

输入音频转换为 16 kHz 单声道,并按 OpenAI Whisper 规则生成 `[1, 80, 3000]` Log-Mel.
Decoder 使用语言 Token、`transcribe` 和 `notimestamps` 提示,首版采用最多 200 Token 的
Greedy Search.

## 限制

模型已完成双 DLA 转换、板端 Smoke Test 和 AISHELL-1 test `7,176` 条正式评测。该 Run
的 NPU CER 为 `45.5935%`,OpenAI 同协议基线 CER 为 `45.8264%`；14 条样例出现重复解码并
达到 Token 上限。LibriSpeech WER、15–30 秒正式样例和噪声鲁棒性仍未验证.

`complete` 只表示指定 Run 无失败或缺失样例,不表示精度达到产品要求。Whisper 可能产生
遗漏、错误文本、繁简体差异或重复幻觉,不应用于未经人工复核的高风险决策.
