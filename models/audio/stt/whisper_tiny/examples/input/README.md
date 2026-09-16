# 样例输入

放置有明确授权的 16 kHz 单声道 WAV.首轮 Smoke Test 至少包含两条内容明显不同的音频,
并覆盖中文、英文、静音和噪声,避免把重复输出或 Stale Buffer 误判为成功.

## OpenAI 官方英文样例

- 文件：`jfk.flac`（不提交到 Git）.
- 固定版本：OpenAI Whisper `v20250625`.
- 下载地址：
  `https://raw.githubusercontent.com/openai/whisper/v20250625/tests/jfk.flac`.
- 用途：双 DLA 英文端到端 Smoke Test 与 OpenAI FP32 Greedy Search 精确文本对比.
- SHA-256：`63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715`.
