# MiniCPM5-2B

> Python 推理代码已迁移,尚未做板端验证.下文报告和示例属于迁移前的 C++/原生 CLI 实现,不能作为当前 Python 实现的精度、性能或 RSS.

面向 Genio 720 的大语言模型适配,采用全 W4A16 量化,支持英文生成、性能统计及精度评价.
使用官方 [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B),许可证 Apache-2.0.
来源记录见 [source_url.txt](models/source_url.txt).

## 执行范围

先在已有编译容器执行 `bash deploy/compile.sh`,完成转换、量化、编译与上传后结束.随后登录开发板,在部署目录执行 `bash compile.sh`,做少量通路检查并保存 README 示例到 `examples/output/`.编译服务器不自动调用板端推理,该阶段不计算或发布精度、性能和峰值 RSS 报告.

正式评价使用独立的 `evaluate.sh`: 容器端只计算已有浮点模型的同协议参考及必要评价输入,资源单独放入已有部署包后,板端运行 `bash evaluate.sh`,统计精度、性能和推理进程峰值 RSS.正式结果只写入 `results/summary.json`,不重新量化或编译.预测、日志及内部指标位于当天日期临时目录.详见[板端规范](../../../../docs/genio_720.md).

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
