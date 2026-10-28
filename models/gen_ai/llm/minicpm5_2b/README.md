# MiniCPM5-2B

> Python 推理已完成固定 16 题 MMLU 板端诊断,尚未完成正式范围评价.下文迁移前报告和示例属于 C++/原生 CLI 实现,不能作为当前 Python 实现的精度、性能或 RSS.

面向 Genio 720 的大语言模型适配,采用全 W4A16 量化,支持英文生成、性能统计及精度评价.
使用官方 [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B),许可证 Apache-2.0.
来源记录见 [source_url.txt](models/source_url.txt).

## 执行范围

先在已有编译容器执行 `bash deploy/compile.sh`,完成转换、量化、编译与上传后结束.随后登录开发板,在部署目录执行 `bash compile.sh`,做少量通路检查并保存 README 示例到 `examples/output/`.编译服务器不自动调用板端推理,该阶段计算并打印子集任务精度,不发布正式报告,不统计性能和峰值 RSS.

正式评价使用独立的 `evaluate.sh`: 容器端只计算已有浮点模型的同协议参考及必要评价输入,资源单独放入已有部署包后,板端运行 `bash evaluate.sh`,统计精度、性能和推理进程峰值 RSS.正式结果只写入 `results/summary.json`,不重新量化或编译.板端预测、日志及内部指标统一位于本模型部署目录的 work/,每次开始推理时覆盖上次中间结果.详见[板端规范](../../../../docs/genio_720.md).

下文已归档结果保留原始模型版本和评价范围,本次入口调整不代表重新实测.

## 运行

模型和数据路径、转换及评价参数统一在 [config.yaml](config.yaml) 配置.
模型资源和数据集提前放在 NAS,运行时不自动下载.

进入编译容器后,在本模型目录执行转换、编译及上传:

```bash
bash deploy/compile.sh
```

上传完成后,登录开发板,进入终端提示的部署目录执行:

```bash
bash compile.sh
```

板端会显示英文示例输入及实际聊天模板,提示 Prefill 阶段,随后实时追加生成文本并显示停止原因.英文按完整词、中文按完整字符刷新,避免显示未完成的 UTF-8 字符.示例最多生成 32 Token,可能只是回答片段,结果保存到 `examples/output/sample_1.txt`.正式评价中的展示输入也会流式打印,独立预热和性能测量关闭文本打印.

正式评价在编译容器中先执行 `bash deploy/evaluate.sh`: MMLU 使用固定 1000 题、57 学科、seed=17 的按学科比例分层抽样,WikiText2 使用完整 raw test.参考计算读取 `evaluation.reference_device`,MMLU 不再默认落到 CPU;CUDA 不可用时明确报错,不自动回退.不运行 W8A16,不重新量化或编译.

将模型资源目录 `build/native/evaluation/mmlu1000/` 的内容单独放入已有板端部署包 `evaluation/mmlu1000/`,按[板端规范](../../../../docs/genio_720.md)核对后在开发板部署目录执行:

```bash
bash evaluate.sh
```

compile.sh 仅生成少量示例,正式指标统一由 evaluate.sh 给出.

MMLU 是无聊天模板、不自动加特殊 Token 的 0-shot 单 Token 选项评分,不生成完整回答,EOS 和生成上限不参与评分.准备时核对原始与转换副本分词器对每题和选项的 Token 一致性,确认空格加 A/B/C/D 在完整提示词后仍为单 Token 延续.参考与板端共用同一份题目 Token、顺序和评分协议.新资源目录不复用旧 `evaluation/full/` 的结果;新生成的题目指纹未与历史清单核对前,不能宣称复现了旧基线.

原始 safetensors 存储为 BF16;当前本地参考将其加载为 FP32 参数计算,并记录存储 dtype、实际参数 dtype、设备及权重文件字节数.它不是官方公布的 FP32 基准成绩,转为 FP32 也不会恢复 BF16 权重舍弃的信息.报告中的 `evaluation_scope=full` 表示完整执行本模型约定范围,MMLU 仍标为固定千题子集,`full_benchmark_evaluated=false`.

W4A16 的误差归因先比较原框架浮点、SDK 转换浮点、同图 W4A16 软件执行和 G720 NPU.先用固定小样本定位 logits 与预测差异,再决定是否运行千题诊断;这些比较不由正式入口自动追加.缺少同版本对照时不将全部掉点归因于量化,不自动判定精度损失合理或验收通过.

## 小批量误差定位,2026-10-26

本轮只使用 seed=17 的固定 16 题,覆盖 13 学科,输入长度 48 至 342 Token;12 题为单块 Prefill,4 题为多块.不运行千题 MMLU、完整 WikiText2、性能或 RSS 评价.官方原始 tokenizer 与实际题目及选项 Token 已在已有新版库中独立核对;旧 SDK 环境不能直接加载原始 tokenizer 的数组形式 BPE merges.

| 同题同 Token 路径 | 答对题数 |
|---|---:|
| 原框架 FP32 参数计算 | 13/16 |
| SDK 浮点 FP32 参数计算 | 13/16 |
| 当前 W4A16 静态图软件执行 | 5/16 |
| 同图软件执行,启用 SDK FP16 计算模拟 | 7/16 |
| 当前 W4A16 DLA,Python 调用真实 NPU | 5/16 |
| 已有 W4A16 Hessian/旋转组合候选,真实 NPU | 8/16 |

原框架与 SDK 浮点的预测 16/16 一致,四选项分数最大绝对差约 0.000061.浮点适配在本批样本上未见明显退化.软件量化与 NPU 虽然同为 5/16,预测仅 4/16 一致;FP16 模拟后与 NPU 预测为 7/16 一致,不能认为软件与硬件已经对齐.单块题也存在差异,不能只归因于跨块 KV.

已有候选与当前模型使用相同官方 revision,但校准、Hessian 和旋转同时不同.候选修正 6 题、退化 3 题,净增加 3 题,仅说明组合配方在本批样本上有改善,不能归因于单项优化或外推整体 MMLU.候选只在日期临时目录保留用于 MMLU 的 Prefill 与配套资源,没有替换当前交付包或默认配置.当前 `results/summary.json` 继续明确保存迁移前历史报告,本轮诊断证据不覆盖该报告.

诊断复用 `tools/llm/evaluate_mmlu.py` 的 `pytorch`、`sdk`、`tflite` 和 `npu` 后端;`sdk` 仅允许最多 16 题,通过 SDK 原生前向驱动读取右侧补齐前的最后有效位置,不发布填充文本的 PPL.`tflite --simulate-fp16` 用于硬件计算精度对照,不改动 DLA.新增 SDK 后端另用本批两题在 CPU 核对入口,预测与已有 SDK 浮点记录一致.过程与逐题四选项分数放日期临时目录,不调用本模型正式 `evaluate.sh`.

## 迁移前结果

以下为已有交付模型的固定子集成绩,不是全量基准.

| 指标 | 本地浮点参考,历史记录标为 FP32 | W4A16 |
|---|---:|---:|
| MMLU,1000 题 | 55.9% | 41.3% |
| WikiText2 PPL,前 128 行 | 40.04 | 48.38 |
| Prefill 速度 | — | 288.67 Token/s |
| Decode 速度 | — | 8.15 Token/s |
| 推理进程峰值 RSS | — | 1152.42 MiB |

以上成绩对应已归档的 `hessian_16_samples` 量化配方及迁移前实现,历史参考运行的完整精度口径仍待原始证据核对,不代表当前默认配方和 Python 实现的结果.55.9% 到 41.3% 是下降 14.6 个百分点,不是 W4A16 的合理损失阈值,也不是官方分数对比.

历史部署大小 2827.29 MB 包含 Prefill/Decode 两个独立 DLA 合计 2292.52 MB 和外置 INT16 embedding 534.77 MB,不能当作单份 INT4 权重体积.输入 Embedding 与 LM Head 各有 267386880 个参数且不共享;估算时不能遗漏 LM Head,也不能把 INT16 embedding 称为 FP16.
结果及适配说明见 [summary.json](results/summary.json) 和 [ADAPTATION.md](ADAPTATION.md).

### 输出误差统计

compile.sh 独立对固定少量输入比较浮点与 NPU 输出,在终端显示 CosSim、RMSE、NRMSE 和实际覆盖范围.普通模型最多选择 5 次输入,不减少原有 compile 通路检查样本数;LLM 比较示例首个 Prefill 块的有效位置,BEVFusion 比较单帧的 NPU 子图.这些结果用于数值检查,不能代替任务精度.全量 evaluate.sh 只计算任务精度、性能和 RSS,不保存或比较全量浮点张量.

参考为普通 NPZ 数组文件,保存在编译机器本地日期临时目录并上传板端 board/output_reference.npz,由下次 compile 覆盖,不使用数据库,不写入 NAS.板端中间结果仍使用本模型 work/.NRMSE 定义为 100 × RMSE / RMS(浮点参考),多输出分别统计.详见[输出误差统计](../../../../docs/output_metrics.md).

## DLA 文件命名

编译和部署包统一使用模型名、精度、层数、阶段、输入 Token 数与上下文长度命名.例如当前 1024 上下文:

- `MiniCPM5-2B_asym4W_sym16A_42layer_prefill_128t_1024c.dla`
- `MiniCPM5-2B_asym4W_sym16A_42layer_decode_1t_1024c.dla`

文件保存在对应的 `1024c/` 目录,加载路径由 `manifest.json` 的 `model_files` 指定.更换上下文或 Prefill 长度时按实际配置命名;历史结果中的旧文件名仅表示当时的产物.

### compile 子集精度检查

compile.sh 使用 100 道覆盖 57 学科的 MMLU 题和 16 个 WikiText2 块,复用正式评价协议.主机执行子集浮点推理,板端执行同一子集 NPU 推理,在终端打印精度和损失,并保存 results/summary_sub.json.正的损失表示 NPU 变差,负值表示改善;Top-1、AP、AR 等使用浮点减 NPU 的百分点,WER 使用 NPU 减浮点的百分点,PPL 使用增量及相对增幅.

普通样本跨完整划分按固定间隔抽取,保留展示及数值检查的必要输入;LFW、ESC-50 和 MMLU 保持折、类别和学科覆盖.同一子集两端通过样本指纹核对;MMLU 固定 seed=17 分层抽样.该报告不代表全量成绩,不改写 results/summary.json.中间数据覆盖保存在本模型 work/,CosSim、RMSE、NRMSE 仍只比较固定少量输入.详见[输出误差统计](../../../../docs/output_metrics.md).
