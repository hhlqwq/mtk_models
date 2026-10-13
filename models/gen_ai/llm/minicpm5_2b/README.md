# MiniCPM5-2B

官方来源: [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B),固定 revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`.
转换输入为官方权重,其他 Model Zoo 仅作选型及交付结构参考.
42 层,hidden=2048,Q/KV Head=16/2,RoPE=5000000,参数量约 2.52B (25.2 亿).

## 当前配置与状态

本项目仅使用全 W4A16: Attention、MLP、LM Head 全部 `asym4W_sym16A`.
当前候选的 128 批校准已完成,量化、编译及板端精度验证待完成.
现有 W4 诊断存在精度退化,不能因推理可运行而认定质量合格.

Prefill=128 Token,上下文=1024,Decode=1 Token.每款只生成两个 DLA.
当前候选: `/tmp/hailongcodex/2026-10-13/minicpm5_w4_fast_v6`.
官方 SDK=2.7.5,Converter=8.16.0,编译目标为 Genio720 MDLA5.3.
Hadamard 权重旋转 seed=0,注意力 mask=-100;只修改独立转换副本.

## 执行流程

在 Ubuntu89 仓库调用 `bash models/gen_ai/llm/minicpm5_2b/deploy/run_native.sh STAGE`.
阶段依次为 `prepare`、`calibrate`、`quantize`、`shape`、`compile`.
设置 `NATIVE_WORK` 为独立候选目录,`MASK_VALUE=-100`,`ROTATE_WEIGHTS=1`,
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
当前官方 FP32 子集 PPL=40.04115;不同模型只比较自身的相对退化.
共享评价入口为 `../../../../tools/llm/evaluate_quality.py`、`../../../../tools/llm/evaluate_board_quality.py`、`../../../../tools/llm/compare_logits.py`.
诊断覆盖四块完整 Prefill 共 512 个输出位置及 Decode 共 64 个后续位置,
单独报告决定首个生成 Token 的 Prefill 最后一行.
Decode mask 的 Runtime 尾部对齐按实际契约补齐,其他输入输出大小严格核对.

正式结果统一报告 Prefill、Decode 的 token/s,不区分中英文.
保留原始 Token、输出、计时日志、版本、文件哈希和实际质量状态.
最终 NAS 目录结构沿用官方模型交付布局;只有通过实际验证的候选才更新交付状态.

MiniCPM5 独立 tokenizer 副本仅转换 BPE merges 序列化格式.校准配置使用单 EOS=130073,运行保留官方停止 Token 1/130073,默认 enable_thinking=false.

## 正式精度指标

首轮结果采用 MMLU 0-shot 固定 1000 题、覆盖 57 学科的 Accuracy,以及 WikiText-2 test 前 128 行的 PPL.共享 `../../../../tools/llm/evaluate_mmlu.py` 固定题目提示、不加入 Chat Template 或额外特殊 Token,使用空格加 A/B/C/D 的单 Token 选项 logits 评分;记录每题分数、答案、预测和完整题数,不得截断或丢题.若输入超过上下文,必须先调整上下文再执行完整测试.

MMLU 官方数据来源为 https://huggingface.co/datasets/cais/mmlu ,固定 revision `c30699e8356da336a370243923dbaf21066bb9fe`,test 文件 SHA256 `74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7`.FP32 与 NPU 使用同一题目 Token 文件及评分协议,报告总题数加权 Accuracy.

首轮 WikiText 评价使用 `evaluate_quality.py --text-rows 128`,继续采用 128 Token 分块、重置上下文、每块评分 127 次下一 Token 的共同协议,记录不足一块的末尾 Token;首轮明确标为固定前 128 行子集,四块诊断不代替首轮结果;后续完整测试才使用 `--full-test`.余弦仅作为误差诊断,不代替上述正式指标.当前尚无新候选的首轮 MMLU/WikiText 结果.

首轮 MMLU 通过 `--limit 1000 --seed 17` 按学科题数比例分层抽样,固定原测试集题目索引并由两个模型共享.结果明确标为首轮子集,不得冒充完整 MMLU Accuracy;后续 `--limit 0` 才执行完整题集.

## 精度对比与修复状态

首轮正式结果见 [first_round_summary.json](results/first_round_summary.json).

| 指标 | 官方 FP32 | 新 W4A16 NPU | 退化 |
|---|---:|---:|---:|
| MMLU Accuracy,固定 1000 题 | 55.9% (559/1000) | 待完成 | 待对比 |
| WikiText PPL,固定前 128 行 | 40.04 | 待完成 | 待对比 |

当前 SDK 浮点模型与官方 FP32 在 4 块、508 个预测位置的平均余弦接近 1,最大 logits 差约 0.000069.此结果只验证已测位置的浮点转换.

旧旋转 W4 候选的 Prefill/Decode 平均余弦: TFLite→NPU 为 0.99967/0.99976,FP32→NPU 为 0.97916/0.98368.该候选的主要误差出现在量化阶段,仍需定位具体层及误差机制;这些诊断值不能替代首轮正式指标.

已实施: Hadamard 旋转及 Hessian/累计层误差优化,扩充长序列校准,完成 128 批校准.旋转方案已有诊断改善,新校准方案的改善效果尚待验证.新 W4 候选尚未完成正式精度对比.

待验证: A16 接口量化范围是否裁剪激活、逐层权重量化误差、长序列校准的改善效果,以及多块 Prefill 的 KV Cache 转移一致性.未获得诊断证据前,这些仅为排查方向.保持全 W4A16,不通过提高权重位宽替代问题定位.

共享工具统一位于仓库 `tools/llm/`;`evaluate_mmlu.py tflite --diagnostic-questions 16` 用于不同输入长度的执行对比,不代替首轮千题 Accuracy.正式性能统一报告 Prefill 和 Decode 的 token/s.

2026-10-13 进一步检查旧旋转 W4 候选的 A16 主接口范围: 4 块数据、86 个接口,约 1.56 亿个数值中仅 54 个超出范围,包含相邻层输出/输入的重复统计.先对照量化 embedding 验证 hidden 的旋转坐标系后再统计.当前没有观察到主接口大面积裁剪,仍不能排除少量异常值的影响或图内算子激活裁剪.详见 [逐层接口报告](results/interface_range_diagnostic.json).下一步检查投影权重量化误差及逐层累积,并对比新校准候选;该报告不替代 MMLU/WikiText 正式成绩.

### 逐层量化误差定位

对旧旋转 W4 候选完成 42 个 Transformer 层及 LM Head 的一致输入诊断.仅使用一块 128 Token;层号从 0 开始.单层误差使用 FP32 hidden 转到实际 R1 坐标系作为量化层输入,累计误差使用上一量化层输出.相对 RMSE 定义为输出误差 RMSE / FP32 参考输出 RMS.

| 层 | 单层相对 RMSE | 累计相对 RMSE |
|---|---:|---:|
| 10 | 0.30% | 0.90% |
| 20 | 0.32% | 1.24% |
| 30 | 0.66% | 2.51% |
| 40 | 1.74% | 7.84% |
| 41 | 3.48% | 13.04% |
| LM Head (42) | 4.24% | 14.95% |

该输入上,误差在 Transformer 层间累积并在后段明显增大;LM Head 单独执行的误差不足以解释累计 logits 误差.残差分支的单层相对误差在 34~38 层约 19%~20%,需要进一步区分注意力、MLP 权重与图内激活误差,尚不能指定某个投影为唯一原因.现有扩充校准和累计层误差优化针对这一累积问题,其效果仍需由新候选实测验证.完整记录见 [逐层误差报告](results/layer_error_diagnostic.json);不将此诊断当作正式精度合格证据.

2026-10-13 按确认切换到常规全 W4A16 量化: 停止本次长耗时重优化,保留校准和历史产物.先生成 Prefill/Decode 两个 DLA 并短测,再进行首轮正式评价;重优化不作为默认部署步骤.

打包从量化成功记录读取实际分片前缀;常规量化的交付清单中 `weight_optimization=null`,不误标为 Hessian.

常规量化未生成优化配置 JSON 时,审核直接读取实际 TFLite: 投影权重须为 INT4,所有图接口须为 INT16,并核对完整层数和文件哈希.可用 `tools/llm/quantize_native.py converter CONFIG -p asym4W_sym16A -tp asym4W_sym16A --audit-only` 恢复审核,不重新量化.
