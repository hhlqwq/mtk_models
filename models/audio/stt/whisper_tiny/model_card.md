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

`complete` 只表示指定 Run 无失败或缺失样例,不表示精度达到产品要求。Whisper 可能产生
遗漏、错误文本、繁简体差异或重复幻觉,不应用于未经人工复核的高风险决策.

## 当前测试结果

以用户上传的 `results/summary.json` 为准,当前待上传。核心指标见 [README](README.md#当前测试结果)。
