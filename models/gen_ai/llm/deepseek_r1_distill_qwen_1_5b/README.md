# DeepSeek-R1-Distill-Qwen-1.5B

官方来源: [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B),固定 revision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.
转换输入为官方权重,其他 Model Zoo 仅作选型及交付结构参考.
28 层,hidden=1536,Q/KV Head=12/2,RoPE=10000.

## 当前配置与状态

本项目仅使用全 W4A16: Attention、MLP、LM Head 全部 `asym4W_sym16A`.
当前候选的 128 批校准已完成,量化、编译及板端精度验证待完成.
现有 W4 诊断存在精度退化,不能因推理可运行而认定质量合格.

Prefill=128 Token,上下文=1024,Decode=1 Token.每款只生成两个 DLA.
当前候选: `/tmp/hailongcodex/2026-10-12/deepseek_w4_rotate_train_v5`.
官方 SDK=2.7.5,Converter=8.16.0,编译目标为 Genio720 MDLA5.3.
Hadamard 权重旋转 seed=0,注意力 mask=-4096;只修改独立转换副本.

## 执行流程

在 Ubuntu89 仓库调用 `bash models/gen_ai/llm/deepseek_r1_distill_qwen_1_5b/deploy/run_native.sh STAGE`.
阶段依次为 `prepare`、`calibrate`、`quantize`、`shape`、`compile`.
设置 `NATIVE_WORK` 为独立候选目录,`MASK_VALUE=-4096`,`ROTATE_WEIGHTS=1`,
`CALIBRATION_BATCHES=128`;使用官方累计层误差优化 JSON 的 `WEIGHT_OPT_CONFIG`.

`prepare` 后调用共享 `../prepare_calibration.py`,按 seed=17 从已校验的 WikiText2
train 固定采样 64 段各 512 Token,保留短对话.记录来源哈希、采样位置和输入哈希,
并使旧校准完成标记失效.测试集不参与校准.
量化采用 Hessian 与累计层误差优化;每阶段保存日志,失败即停止.
共享 `../quantize_native.py` 传播官方 SDK 异常,不修改安装文件.
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
共享评价入口为 `../evaluate_quality.py`、`../evaluate_board_quality.py`、`../compare_logits.py`.
诊断覆盖四块完整 Prefill 共 512 个输出位置及 Decode 共 64 个后续位置,
单独报告决定首个生成 Token 的 Prefill 最后一行.
Decode mask 的 Runtime 尾部对齐按实际契约补齐,其他输入输出大小严格核对.

正式结果统一报告 Prefill、Decode 的 token/s,不区分中英文.
保留原始 Token、输出、计时日志、版本、文件哈希和实际质量状态.
最终 NAS 目录结构沿用官方模型交付布局;只有通过实际验证的候选才更新交付状态.

## 正式精度指标

正式结果采用 MMLU 0-shot 完整 57 学科测试集 Accuracy,以及完整 WikiText-2 test 的 PPL.共享 `../evaluate_mmlu.py` 固定题目提示、不加入 Chat Template 或额外特殊 Token,使用空格加 A/B/C/D 的单 Token 选项 logits 评分;记录每题分数、答案、预测和完整题数,不得截断或丢题.若输入超过上下文,必须先调整上下文再执行完整测试.

MMLU 官方数据来源为 https://huggingface.co/datasets/cais/mmlu ,固定 revision `c30699e8356da336a370243923dbaf21066bb9fe`,test 文件 SHA256 `74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7`.FP32 与 NPU 使用同一题目 Token 文件及评分协议,报告总题数加权 Accuracy.

WikiText 评价使用 `evaluate_quality.py --full-test`,继续采用 128 Token 分块、重置上下文、每块评分 127 次下一 Token 的共同协议,记录不足一块的末尾 Token;不得将此前前 128 行或四块诊断 PPL 标为完整测试结果.余弦仅作为误差诊断,不代替上述正式指标.当前尚无新候选的完整 MMLU/WikiText 结果.
