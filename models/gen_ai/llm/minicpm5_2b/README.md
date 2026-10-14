# MiniCPM5-2B

官方来源: [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B),固定 revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`,Apache-2.0.使用官方权重作为转换输入,其他 Model Zoo 仅作选型和交付结构参考.42 层,hidden=2048,Q/KV Head=16/2,RoPE=5000000,参数量约 2.52B (25.2 亿).

## 当前实测

当前交付模型已完成全 W4A16 量化、Prefill/Decode 双 DLA 编译、Genio720 MDLA 板端性能及首轮精度评价.与官方 FP32 仍有明显差距,保持 `quality_qualified=false`.结果表只展示当前交付模型,优化过程见后文.

Attention、MLP、LM Head 全部 `asym4W_sym16A`,没有 W8 或混合精度.使用 Hadamard 旋转 seed=0、mask=-100,仅修改独立转换副本.官方 SDK=2.7.5,Converter=8.16.0,编译目标 MDLA5.3.

| 首轮指标 | 官方 FP32 基线 | 当前交付 W4A16 |
|---|---:|---:|
| MMLU Accuracy,0-shot 固定 1000 题 | 55.9% (559/1000) | 41.3% (413/1000) |
| WikiText2 PPL,测试集前 128 行 | 40.0411 | 48.3821 |

当前 MMLU 相对 FP32 下降 14.6 个百分点,WikiText PPL 相对 FP32 增加 20.83%.这是首轮固定子集成绩,不代表完整测试集.当前候选完成了相同协议的完整首轮评测;先前的小范围诊断不能用于选出另一款“最好”交付模型.

Prefill=128 Token,Decode=1 Token,上下文=1024.只有 `1024c/prompt.dla` 和 `1024c/decode.dla` 两个 DLA.固定 128 Token 输入、最多 128 Token 输出,Prefill **288.67 token/s**,Decode **8.15 token/s**.使用 MDLA,没有 CPU 回退.性能输入生成 128 Token,未到 EOS;计时不代表自由生成质量通过.

结果及哈希: [首轮汇总](results/first_round_summary.json)、[性能记录](results/native_summary.json)、[Prefill 误差](results/logits_summary.json)、[Decode 误差](results/decode_logits_summary.json).

## 模型适配与优化过程

1. **固定官方输入与部署配置.** 使用固定 revision 的官方权重,保留原始文件及来源哈希.先确定 Prefill=128、Decode=1、上下文=1024,后续始终保持全 W4A16 和两个 DLA.独立 tokenizer 副本只调整 BPE merges 的序列化格式,不改词表含义;运行保留官方停止 Token.
2. **验证浮点转换.** 在独立转换副本中适配模型结构、RoPE 和注意力掩码,按官方 FP32 -> SDK 浮点逐项核对输入和 logits.四块、508 个预测位置的平均余弦接近 1,最大 logits 差约 0.000069.由此确认已测位置的浮点转换一致,再进入量化.
3. **准备训练集校准并生成常规量化模型.** 按 seed=17 从 WikiText2 train 固定采样 64 段各 512 Token,保留短对话校准数据,完成 128 批校准.记录文本来源、采样位置和输入哈希,测试集不参与校准.应用 Hadamard 旋转后先执行常规全 W4A16 量化,编译两个 DLA 并进行同协议板端评价;发现明显精度退化后进入误差排查.
4. **分段定位掉点.** 对照 FP32、量化 TFLite 和 NPU,主要损失发生在量化阶段.逐层诊断显示误差在后段累积,LM Head 单独误差不足以解释整模损失.主接口范围检查没有发现大面积 A16 裁剪,但不能排除图内激活误差.相关过程记录见 [逐层报告](results/layer_error_diagnostic.json) 和 [范围报告](results/interface_range_diagnostic.json);它们不作为当前模型的成绩展示.
5. **实施有时长上限的量化修复.** 先固定同一训练校准样本和旋转矩阵,对单层比较常规量化与 Hessian,确认权重量化误差可以改善.随后每层选用 16 个训练校准样本进行 Hessian,先处理首层、后段层和 LM Head,再复用这些结果并处理剩余层,最终覆盖全部 43 分片.关闭耗时累计层误差优化,不训练或微调原始 FP32 权重.SDK 量化合计约 15 分 14 秒,不含初始化、导出和编译.
6. **合并双图并完成板端验证.** 分片 A16 hidden 尺度不同时,显式使用官方 `Requantize`,仅接受同形状 INT16 单尺度接口并保留 Builder 检查.修复 SDK 临时目录清理导致的导出误报.审核实际权重与接口精度后编译两个 DLA,完成 Prefill/Decode 诊断、统一性能、固定 1000 题 MMLU 和前 128 行 WikiText 评价.按当前实测结果打包,复制到 NAS 后逐文件核对 SHA256.

过程记录保留每一步的配置、依据和处理结论,当前结果表不保留实验编号或历史候选成绩.

## 当前精度分析

| 当前模型平均余弦 | Prefill,508 个评分位置 | Decode,64 个位置 |
|---|---:|---:|
| FP32 -> TFLite | 0.97083 | 0.97338 |
| FP32 -> NPU | 0.97115 | 0.97385 |
| TFLite -> NPU | 0.99955 | 0.99971 |

Prefill 完整 512 个输出位置的 FP32 -> NPU 平均余弦为 0.97105.四块诊断用于定位误差,不替代 MMLU Accuracy 与 WikiText PPL.主要损失已在量化阶段产生,板端新增误差较小.

修复已完成整模首轮实测.剩余 14.6 个百分点 Accuracy 差距仍未解决;需继续区分投影误差、图内激活误差及后段累积,不能用接近 1 的 TFLite/NPU 余弦宣称精度合格.

逐题分析中,FP32 正确而 NPU 错误 252 题,反向改善 106 题.不超过 128 Token 的 756 题只需一次 Prefill,FP32 正确 440 题、NPU 正确 323 题;因此剩余退化不能只归因于多块 Prefill 的 KV Cache 转移.长于 512 Token 的首轮题目仅 9 道,不据此推断长上下文精度.详见 [逐题误差统计](results/mmlu_error_analysis.json).

## 统一评价协议

MMLU 使用 [cais/mmlu](https://huggingface.co/datasets/cais/mmlu),固定 revision `c30699e8356da336a370243923dbaf21066bb9fe`,test SHA256 `74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7`.0-shot,seed=17,按 57 学科题数比例固定抽样 1000 题.不加 Chat Template 或额外特殊 Token,取空格加 A/B/C/D 的单 Token logits 评分.最长题目 968 Token,没有截断或丢题.FP32/NPU 的题目 ID、原始索引、学科、答案和 cases SHA 已逐项核对.板端千题评分约 10 分 49 秒.

WikiText2 使用 test 前 128 行原始文本,不加特殊 Token,每块 128 Token、重置上下文、评分 127 次下一 Token.两端原始文本 SHA、协议、58 块、7366 个评分位置与舍弃尾部 33 Token 对齐.板端评分约 36 秒.固定子集结果不冒充完整 PPL.

## 执行与产物

公共脚本统一位于仓库 `tools/llm/`.在 Ubuntu89 仓库调用 `bash models/gen_ai/llm/minicpm5_2b/deploy/run_native.sh STAGE`,阶段为 `prepare`、`calibrate`、`quantize`、`shape`、`compile`.

常规入口默认 `WEIGHT_OPTIMIZATION=none`,不默认开启长耗时优化.当前交付模型使用归档的 16 样本、分阶段 Hessian 配方;直接运行默认入口不会生成相同模型.量化记录须保存实际优化分片、样本数、累计层误差开关和旋转配置.测试集不参与校准.

量化后审核实际 TFLite 的 295 个投影权重为 INT4、接口为 INT16及完整 43 分片哈希.可用 `tools/llm/quantize_native.py --audit-only converter CONFIG -p asym4W_sym16A -tp asym4W_sym16A` 单独审核.打包读取成功记录中的实际前缀与优化配置;MMLU 使用 `1024c/prompt.dla` 并核对契约和清单.

独立 tokenizer 副本仅转换 BPE merges 序列化格式.校准配置使用单 EOS=130073,运行保留官方停止 Token 1/130073,默认 enable_thinking=false.

工具归档目录: `/data/users/hailong.he/data/MTKG720/GAI_Toolkit/MiniCPM5-2B/`,明确标为项目适配脚本,不是 MTK 官方 MiniCPM5 专用 Toolkit.

最终 NAS 根目录采用用户指定的 G720 外部 `models/` 目录,模型包相对路径为 `gen_ai/llm/minicpm5_2b/minicpm5-2b/`.保留该模型目录已有的来源记录,复制后逐文件核对 SHA256,记录在模型包的 `delivery_verification.json` 中.服务器模型可运行,精度尚未通过质量验收.
