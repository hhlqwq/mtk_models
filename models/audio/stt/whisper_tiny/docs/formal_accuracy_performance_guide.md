# Whisper-Tiny 正式精度与性能评测指南

本文档只提供可复现评测链路，不包含未执行的 WER、CER 或性能结果。数据集下载与正式测试
由用户执行；脚本不会联网下载数据，也不会修改原始数据。

## 1. 官方数据

| 数据集 | 正式子集 | 来源页面 | 下载地址 | 许可证 | 官方体量 |
|---|---|---|---|---|---:|
| LibriSpeech | `test-clean` | `https://www.openslr.org/12/` | `https://www.openslr.org/resources/12/test-clean.tar.gz` | CC BY 4.0 | 346 MB |
| AISHELL-1 | `test` | `https://www.openslr.org/33/` | `https://openslr.trmal.net/resources/33/data_aishell.tgz` | Apache License 2.0 | 15 GB |

AISHELL-1 使用上述 OpenSLR 镜像下载，官方 SLR33 页面继续作为数据集来源与
许可证依据。下载完成后先使用 OpenSLR 提供的 MD5 文件验证，再记录实测 SHA-256。建议把压缩包和
解压目录放在 89 的
`/data/users/hailong.he/nas_smb/Datasets/open_source/raw/`；该 NAS 路径已经只读挂载
到 `hhl_g720_8011`。也可以放到其他目录，但必须通过 `DATASET_ROOT` 和 `ARCHIVE` 显式传入。

项目只评测不超过 30 秒的音频。清单构建器会把超过 30 秒的条目写入
`excluded_over_30s.jsonl`，不会静默截断后混入正式 WER/CER；报告必须同时披露排除数量。

## 2. 一键正式评测入口

正式入口必须在 Ubuntu89 宿主机执行,不要进入 `hhl_g720_8011` 容器。它会自行调用容器
环境和板端,依次完成两套数据的交叉编译、数据与框架基线准备、全部样本板端推理及联合
报告：

```bash
cd /data/users/hailong.he/github/mtk_models
bash models/audio/stt/whisper_tiny/deploy/run_all_formal_evaluations.sh
```

双数据集入口会在开始前校验 AISHELL-1 与 LibriSpeech 的目录和压缩包,只编译一次板端程序,
并顺序执行两套 Run。已有 `complete`、失败 0、缺失 0 的 Run 默认直接复用；设置
`REUSE_COMPLETE_RUNS=0` 才会强制重新执行。只有两套 Run 都完整时才生成联合报告并输出
“两套正式评测全部完成”。

`run_formal_evaluation.sh` 是单数据集恢复入口；`prepare_accuracy.sh`、
`run_accuracy_board.sh` 和 `summarize_accuracy.sh` 是更底层的阶段脚本。它们仅用于定位故障
或断点恢复,不能代表双数据集全部完成。容器生成文件的所有权会恢复为当前宿主用户。

生成的 `whisper_board_eval` 会在一次进程内持久加载 Encoder/Decoder DLA，并对 JSONL
清单逐条推理。输出每完成一条就落盘；重复使用同一个输出路径时会跳过已有 `status=ok`
的样例，从而实现断点续跑。

## 3. 当前补跑命令

AISHELL-1 Run `20260918_aishell1_test_fp16_v1` 已完成。使用下列命令时,双数据集入口会复用
该 Run,只执行尚未完成的 LibriSpeech,再生成联合报告：

```bash
cd /data/users/hailong.he/github/mtk_models
export EVAL_DATE=20260921
export AISHELL_RUN_ID=20260918_aishell1_test_fp16_v1
bash models/audio/stt/whisper_tiny/deploy/run_all_formal_evaluations.sh
```

跨天恢复时继续使用相同的 `EVAL_DATE`,避免创建新的 LibriSpeech Run 目录。

## 4. 单数据集故障恢复

以下命令不是完整正式评测入口,只在对应数据集失败后用于单独恢复。

### LibriSpeech test-clean

89 上的 `test-clean` 已解压到
`/data/users/hailong.he/nas_smb/Datasets/open_source/raw/LibriSpeech/test-clean/LibriSpeech/test-clean/`：

```bash
export DATASET=librispeech
export RUN_ID=20260918_librispeech_test_clean_fp16_v1
export DATASET_ROOT=/data/users/hailong.he/nas_smb/Datasets/open_source/raw/LibriSpeech/test-clean/LibriSpeech/test-clean
export ARCHIVE=/data/users/hailong.he/nas_smb/Datasets/open_source/raw/LibriSpeech/test-clean.tar.gz

bash models/audio/stt/whisper_tiny/deploy/run_formal_evaluation.sh
```

### AISHELL-1 test

89 上已单独解压官方 test 的 20 位说话人、7,176 条 WAV；目录同时包含完整转录文件：

```bash
export DATASET=aishell1
export RUN_ID=20260918_aishell1_test_fp16_v1
export DATASET_ROOT=/data/users/hailong.he/nas_smb/Datasets/open_source/raw/Aishell/test
export ARCHIVE=/data/users/hailong.he/nas_smb/Datasets/open_source/raw/Aishell/data_aishell.tgz

bash models/audio/stt/whisper_tiny/deploy/run_formal_evaluation.sh
```

如果 4090 显存不足，可降低 `REFERENCE_BATCH_SIZE`。一键正式评测必须生成 OpenAI 框架
基线,不允许设置 `REFERENCE_DEVICE=none`；框架基线用于统计 NPU 与 OpenAI 输出的规范化
文本、Token 完全一致率。

## 5. 已完成正式 Run

AISHELL-1 Run `20260918_aishell1_test_fp16_v1` 已完成 `7,176/7,176` 条样例,失败与缺失均为
0。NPU CER 为 `45.5935%`,OpenAI 同协议 CER 为 `45.8264%`；详细一致性、截断说明和性能
数据见 [`accuracy.md`](accuracy.md)、[`benchmark.md`](benchmark.md) 与
[`formal_eval_20260918_aishell1.json`](formal_eval_20260918_aishell1.json).

LibriSpeech `test-clean` 尚无正式 `summary.json`,不得填写或推断 WER.

## 6. 指标口径

- 英文：OpenAI `EnglishTextNormalizer` 后按单词计算 WER。
- 中文：OpenAI `BasicTextNormalizer` 后移除空白，按 Unicode 字符计算 CER。
- 精度：报告替换、删除、插入、参考单元总数、WER/CER 和规范化文本完全一致率。
- 一致性：同时报告板端 NPU 与 OpenAI 框架基线的文本和 Token 完全一致数量。
- 性能：只统计 `NeuronRuntime_inference` 的 Encoder、首 Token、Decoder、NPU 总耗时、
  Tokens/s 和 NPU RTF；89 的音频读取、Log-Mel 和 Mel 写入耗时单独记录在
  `preprocess_metrics.jsonl`,不与板端 NPU 时间混算。
- 分组：`≤5 s`、`5–15 s`、`15–25 s`、`25–30 s`，分别输出 Mean/P50/P90/P95。
- 内存：板端程序逐条记录进程 `ru_maxrss`，外层 `/usr/bin/time -v` 保存独立原始记录。

## 7. 运行目录与验收

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
