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

当前只有实现框架,没有转换或板端证据.不得将静态检查、Encoder 单图编译或 NCC
执行计划检查描述为完整 ASR 部署.Whisper 可能产生遗漏、错误文本或幻觉,不应用于未经人工
复核的高风险决策.
