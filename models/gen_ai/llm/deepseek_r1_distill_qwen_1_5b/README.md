# DeepSeek-R1-Distill-Qwen-1.5B

编译与转换步骤统一在 Docker 容器内执行: 先在仓库根目录运行 `bash docker/enter_container.sh`,再进入本模型目录运行原有 `deploy/run.sh` 或 `deploy/run_native.sh` 入口.模型脚本不再调用 Docker 或在主机和容器之间复制文件; 上传完成后,板端入口仍在开发板执行.

官方来源: [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B),固定 revision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.
转换输入为官方权重,其他 Model Zoo 仅作选型及交付结构参考.
28 层,hidden=1536,Q/KV Head=12/2,RoPE=10000.

## 当前配置与状态

本项目仅使用全 W4A16: Attention、MLP、LM Head 全部 `asym4W_sym16A`.
当前候选的 128 批校准已完成,量化、编译及板端精度验证待完成.
现有 W4 诊断存在精度退化,不能因推理可运行而认定质量合格.

Prefill=128 Token,上下文=1024,Decode=1 Token.每款只生成两个 DLA.
当前候选编号: `deepseek_w4_rotate_train_v5`.实际工作目录通过 `NATIVE_WORK` 指定.
官方 SDK=2.7.5,Converter=8.16.0,编译目标为 Genio720 MDLA5.3.
Hadamard 权重旋转 seed=0,注意力 mask=-4096;只修改独立转换副本.

## 执行流程

在编译容器的项目根目录调用 `bash models/gen_ai/llm/deepseek_r1_distill_qwen_1_5b/deploy/run_native.sh STAGE`.
阶段依次为 `prepare`、`calibrate`、`quantize`、`shape`、`compile`.
设置 `NATIVE_WORK` 为独立候选目录,`MASK_VALUE=-4096`,`ROTATE_WEIGHTS=1`,
`CALIBRATION_BATCHES=128`;默认 `WEIGHT_OPTIMIZATION=none`,清空 `WEIGHT_OPT_CONFIG`,复用已完成的校准数据.

`prepare` 后调用共享 `../../../../tools/llm/prepare_calibration.py`,按 seed=17 从已校验的 WikiText2
train 固定采样 64 段各 512 Token,保留短对话.记录来源哈希、采样位置和输入哈希,
并使旧校准完成标记失效.测试集不参与校准.
常规量化默认关闭 Hessian 与累计层误差优化,保持全 W4A16.只有短板端验证和正式对比明确需要时,才显式启用 `WEIGHT_OPTIMIZATION=hessian` 与优化 JSON;每阶段保存日志,失败即停止.
共享 `../../../../tools/llm/quantize_native.py` 传播官方 SDK 异常,不修改安装文件.
量化后逐层核对实际权重配置并生成 `weight_precision_audit.json`.

`deploy/host/package_native.py` 要求独立空输出目录,核对权重精度和实际分片数量,
将核对记录及源文件哈希写入交付清单.编译成功不能代替板端验证.

## 精度排查与评价

按官方 FP32 -> SDK 浮点 -> 量化 TFLite -> 板端 MDLA 定位误差:

- 先核对 tokenizer、特殊 Token、RoPE、注意力掩码、模型权重和浮点转换一致性.
- 浮点一致后,核对校准代表性、权重旋转、量化范围及 Hessian 优化误差.
- 主机量化与 NPU 差异单独检查输入输出量化尺度、KV Cache、图编译和运行契约.
- 不通过提高权重位宽掩盖精度问题;不合格候选保持未通过状态.

统一评价采用 WikiText2 test 前 128 行原始文本,不额外加入特殊 Token,
每块 128 Token、重置上下文、评分 127 次下一 Token,舍弃并记录不足一块的尾部.
当前官方 FP32 子集 PPL=118.70850;不同模型只比较自身的相对退化.
共享评价入口为 `../../../../tools/llm/evaluate_quality.py`、`../../../../tools/llm/evaluate_board_quality.py`、`../../../../tools/llm/compare_logits.py`.
诊断覆盖四块完整 Prefill 共 512 个输出位置及 Decode 共 64 个后续位置,
单独报告决定首个生成 Token 的 Prefill 最后一行.
Decode mask 的 Runtime 尾部对齐按实际契约补齐,其他输入输出大小严格核对.

正式结果统一报告 Prefill、Decode 的 token/s,不区分中英文.
保留原始 Token、输出、计时日志、版本、文件哈希和实际质量状态.
最终 NAS 目录结构沿用官方模型交付布局;只有通过实际验证的候选才更新交付状态.

## 正式精度指标

首轮结果采用 MMLU 0-shot 固定 1000 题、覆盖 57 学科的 Accuracy,以及 WikiText-2 test 前 128 行的 PPL.共享 `../../../../tools/llm/evaluate_mmlu.py` 固定题目提示、不加入 Chat Template 或额外特殊 Token,使用空格加 A/B/C/D 的单 Token 选项 logits 评分;记录每题分数、答案、预测和完整题数,不得截断或丢题.若输入超过上下文,必须先调整上下文再执行完整测试.

MMLU 官方数据来源为 https://huggingface.co/datasets/cais/mmlu ,固定 revision `c30699e8356da336a370243923dbaf21066bb9fe`,test 文件 SHA256 `74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7`.FP32 与 NPU 使用同一题目 Token 文件及评分协议,报告总题数加权 Accuracy.

首轮 WikiText 评价使用 `evaluate_quality.py --text-rows 128`,继续采用 128 Token 分块、重置上下文、每块评分 127 次下一 Token 的共同协议,记录不足一块的末尾 Token;首轮明确标为固定前 128 行子集,四块诊断不代替首轮结果;后续完整测试才使用 `--full-test`.余弦仅作为误差诊断,不代替上述正式指标.当前尚无新候选的首轮 MMLU/WikiText 结果.

首轮 MMLU 通过 `--limit 1000 --seed 17` 按学科题数比例分层抽样,固定原测试集题目索引并由两个模型共享.结果明确标为首轮子集,不得冒充完整 MMLU Accuracy;后续 `--limit 0` 才执行完整题集.

## 精度对比与修复状态

首轮正式结果见 [first_round_summary.json](results/first_round_summary.json).

| 指标 | 官方 FP32 | 新 W4A16 NPU | 退化 |
|---|---:|---:|---:|
| MMLU Accuracy,固定 1000 题 | 37.2% (372/1000) | 待完成 | 待对比 |
| WikiText PPL,固定前 128 行 | 118.71 | 待完成 | 待对比 |

已确认注意力掩码会影响浮点转换一致性.当前 mask=-4096 的 SDK 浮点模型与官方 FP32 在 4 块、508 个预测位置的平均余弦接近 1,最大 logits 差约 0.000134.此结果只验证已测位置的浮点转换.

旧掩码修复 W4 候选在相同 4 块诊断数据上,FP32 PPL=93.38,NPU PPL=132.17,增加约 41.5%;Prefill/Decode 的 FP32→NPU 平均余弦为 0.8941/0.8993.掩码修复没有消除剩余量化误差,这些诊断值不能替代首轮正式指标.

已实施: 修正掩码,启用 Hadamard 旋转及 Hessian/累计层误差优化,扩充长序列校准,完成 128 批校准.新 W4 候选仍待量化、编译和板端评价完成,尚未证明修复后的正式精度合格.

待验证: A16 接口量化范围是否裁剪激活、逐层权重量化误差、长序列校准的改善效果,以及多块 Prefill 的 KV Cache 转移一致性.未获得诊断证据前,这些仅为排查方向.保持全 W4A16,不通过提高权重位宽替代问题定位.

共享工具统一位于仓库 `tools/llm/`;`evaluate_mmlu.py tflite --diagnostic-questions 16` 用于不同输入长度的执行对比,不代替首轮千题 Accuracy.正式性能统一报告 Prefill 和 Decode 的 token/s.

2026-10-13 按确认切换到常规全 W4A16 量化: 停止本次长耗时重优化,保留校准和历史产物.先生成 Prefill/Decode 两个 DLA 并短测,再进行首轮正式评价;重优化不作为默认部署步骤.

## 模型资源路径

原始权重、ONNX、量化模型、DLA 和配套运行参数使用仓库外的模型存储目录.先配置 `configs/storage.local.json` 的 `models_root` 和 `dataset_root`,完整内部路径仅保存本地.本模型的外部目录为 `models_root/gen_ai/llm/deepseek_r1_distill_qwen_1_5b`;编译入口自动读取配置,转换产物也写入此目录.仓库中的 `models/` 只保留源码、文档和公开来源记录.板端继续使用上传的部署目录.
