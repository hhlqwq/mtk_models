# Whisper-Tiny 正式精度与性能评测指南

本文档只提供可复现评测链路，不包含未执行的 WER、CER 或性能结果。数据集下载与正式测试
由用户执行；脚本不会联网下载数据，也不会修改原始数据。

## 1. 官方数据

| 数据集 | 正式子集 | 官方地址 | 许可证 | 官方体量 |
|---|---|---|---|---:|
| LibriSpeech | `test-clean` | `https://www.openslr.org/resources/12/test-clean.tar.gz` | CC BY 4.0 | 346 MB |
| AISHELL-1 | `test` | `https://www.openslr.org/resources/33/data_aishell.tgz` | Apache License 2.0 | 15 GB |

下载完成后先使用 OpenSLR 同目录提供的 MD5 文件验证，再记录实测 SHA-256。建议把压缩包和
解压目录放在 89 的
`/data/users/hailong.he/nas_smb/Datasets/open_source/raw/speech/`；该 NAS 路径已经只读挂载
到 `hhl_g720_8011`。也可以放到其他目录，但必须通过 `DATASET_ROOT` 和 `ARCHIVE` 显式传入。

项目只评测不超过 30 秒的音频。清单构建器会把超过 30 秒的条目写入
`excluded_over_30s.jsonl`，不会静默截断后混入正式 WER/CER；报告必须同时披露排除数量。

## 2. 构建批量评测程序

在 Ubuntu89 的项目根目录执行：

```bash
cd /data/users/hailong.he/github/mtk_models
bash models/audio/stt/whisper_tiny/deploy/build_board_cpp.sh
```

生成的 `whisper_board_eval` 会在一次进程内持久加载 Encoder/Decoder DLA，并对 JSONL
清单逐条推理。输出每完成一条就落盘；重复使用同一个输出路径时会跳过已有 `status=ok`
的样例，从而实现断点续跑。

## 3. LibriSpeech test-clean

假设解压后存在 `LibriSpeech/test-clean/`：

```bash
export DATASET=librispeech
export RUN_ID=20260917_librispeech_test_clean_fp16_v1
export DATASET_ROOT=/data/users/hailong.he/nas_smb/Datasets/open_source/raw/speech/LibriSpeech/test-clean
export ARCHIVE=/data/users/hailong.he/nas_smb/Datasets/open_source/raw/speech/test-clean.tar.gz

bash models/audio/stt/whisper_tiny/deploy/prepare_accuracy.sh
bash models/audio/stt/whisper_tiny/deploy/run_accuracy_board.sh
bash models/audio/stt/whisper_tiny/deploy/summarize_accuracy.sh
```

## 4. AISHELL-1 test

假设解压根目录为 `data_aishell/`：

```bash
export DATASET=aishell1
export RUN_ID=20260917_aishell1_test_fp16_v1
export DATASET_ROOT=/data/users/hailong.he/nas_smb/Datasets/open_source/raw/speech/data_aishell
export ARCHIVE=/data/users/hailong.he/nas_smb/Datasets/open_source/raw/speech/data_aishell.tgz

bash models/audio/stt/whisper_tiny/deploy/prepare_accuracy.sh
bash models/audio/stt/whisper_tiny/deploy/run_accuracy_board.sh
bash models/audio/stt/whisper_tiny/deploy/summarize_accuracy.sh
```

如果 4090 显存不足，可降低 `REFERENCE_BATCH_SIZE`；如果只想先生成 Mel 而暂不跑 OpenAI
框架基线，可设置 `REFERENCE_DEVICE=none`。正式交付时仍应补跑框架基线，用于统计 NPU 与
OpenAI 输出的规范化文本、Token 完全一致率。

## 5. 指标口径

- 英文：OpenAI `EnglishTextNormalizer` 后按单词计算 WER。
- 中文：OpenAI `BasicTextNormalizer` 后移除空白，按 Unicode 字符计算 CER。
- 精度：报告替换、删除、插入、参考单元总数、WER/CER 和规范化文本完全一致率。
- 一致性：同时报告板端 NPU 与 OpenAI 框架基线的文本和 Token 完全一致数量。
- 性能：只统计 `NeuronRuntime_inference` 的 Encoder、首 Token、Decoder、NPU 总耗时、
  Tokens/s 和 NPU RTF；89 的音频读取、Log-Mel 和 Mel 写入耗时单独记录在
  `preprocess_metrics.jsonl`,不与板端 NPU 时间混算。
- 分组：`≤5 s`、`5–15 s`、`15–25 s`、`25–30 s`，分别输出 Mean/P50/P90/P95。
- 内存：板端程序逐条记录进程 `ru_maxrss`，外层 `/usr/bin/time -v` 保存独立原始记录。

## 6. 运行目录与验收

89 的全部中间文件位于 `.eval/whisper_tiny/<run_id>/`，该目录被 Git 忽略。92 的对应目录为
`/root/hailong.he/whisper_tiny/eval/<run_id>/`。禁止复用不同数据集的 Run ID。

每个正式 Run 至少应包含：

- `dataset_manifest.json`、`source_manifest.jsonl`、`excluded_over_30s.jsonl`；
- `input_manifest.json`、`board_manifest.tsv`、`decode_config.txt`；
- `reference_predictions.jsonl`；
- `board/board_predictions.jsonl`、`board.log`、`resource_usage.txt`；
- `report/summary.json`、`worst_samples.jsonl`、`report.md`。

只有 `summary.json` 的 `status` 为 `complete`、失败和缺失样例均为 0，才能把对应指标写入
正式精度/性能文档。脚本发现不完整结果时默认返回非零；修复问题后用相同 Run ID 重跑即可。
