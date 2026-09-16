# 样例输出

运行证据写入 `runs/<run_id>/`,至少包含转写文本、Token、逐阶段耗时、Runtime 版本、
模型与输入哈希、退出状态和原始日志.

2026-09-16 已生成两个本地忽略的运行目录：

- `runs/20260916_jfk_fp32_v1/`：23 个 Token 与 OpenAI FP32 基线完全一致.
- `runs/20260916_silence_fp32_v1/`：1 个 Token 与 OpenAI FP32 基线完全一致,用于排除
  JFK Stale Buffer.

可提交的汇总证据见 `../../docs/board_smoke_20260916.json`.
