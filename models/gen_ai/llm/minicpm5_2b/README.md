# MiniCPM5-2B

官方来源: [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B),固定 revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`,Apache-2.0.使用官方权重作为转换输入,其他 Model Zoo 仅作选型和交付结构参考.42 层,hidden=2048,Q/KV Head=16/2,RoPE=5000000,参数量约 2.52B (25.2 亿).

## 当前实测

2026-10-14 的 `minicpm5_w4_light_hessian_v8` 已完成全 W4A16 量化、Prefill/Decode 双 DLA 编译、Genio720 MDLA 板端性能及首轮精度评价.相比常规 W4 候选明显改善,但与官方 FP32 仍有差距,保持 `quality_qualified=false`.

Attention、MLP、LM Head 全部 `asym4W_sym16A`,没有 W8 或混合精度.使用 Hadamard 旋转 seed=0、mask=-100,仅修改独立转换副本.官方 SDK=2.7.5,Converter=8.16.0,编译目标 MDLA5.3.

| 首轮指标 | 官方 FP32 | 常规 W4 v6 | 当前 W4 v8 |
|---|---:|---:|---:|
| MMLU Accuracy,0-shot 固定 1000 题 | 55.9% (559/1000) | 34.8% (348/1000) | 41.3% (413/1000) |
| WikiText2 PPL,测试集前 128 行 | 40.0411 | 75.6491 | 48.3821 |

当前 MMLU 相对 FP32 下降 14.6 个百分点,比 v6 提升 6.5 个百分点.WikiText PPL 相对 FP32 增加 20.83%,比 v6 降低 36.04%.这是首轮固定子集成绩,不代表完整测试集.

Prefill=128 Token,Decode=1 Token,上下文=1024.只有 `1024c/prompt.dla` 和 `1024c/decode.dla` 两个 DLA.固定 128 Token 输入、最多 128 Token 输出,Prefill **288.67 token/s**,Decode **8.15 token/s**.使用 MDLA,没有 CPU 回退.性能输入生成 128 Token,未到 EOS;计时不代表自由生成质量通过.

结果及哈希: [首轮汇总](results/first_round_summary.json)、[性能记录](results/native_summary.json)、[Prefill 误差](results/logits_summary.json)、[Decode 误差](results/decode_logits_summary.json).

## 精度下降原因与修复

按官方 FP32 -> SDK 浮点 -> 量化 TFLite -> NPU 分段比较.四块、508 个预测位置的 SDK 浮点与官方 FP32 平均余弦接近 1,最大 logits 差约 0.000069;此结论限于已测位置.

| 当前 v8 平均余弦 | Prefill,508 个评分位置 | Decode,64 个位置 |
|---|---:|---:|
| FP32 -> TFLite | 0.97083 | 0.97338 |
| FP32 -> NPU | 0.97115 | 0.97385 |
| TFLite -> NPU | 0.99955 | 0.99971 |

Prefill 完整 512 个输出位置的 FP32 -> NPU 平均余弦为 0.97105.四块诊断用于定位误差,不替代 MMLU Accuracy 与 WikiText PPL.主要损失已在量化阶段产生,板端新增误差较小.

已确认和实施的处理:

- 旧旋转 W4 逐层诊断显示后段误差累积: 第 41 层累计相对 RMSE 13.04%,LM Head 单层 4.24%、累计 14.95%.仅优化 LM Head 不能解决全部损失.详见 [逐层报告](results/layer_error_diagnostic.json).
- A16 主接口四块约 1.56 亿个数值中仅 54 个超范围,含相邻接口重复统计.未观察到主接口大面积裁剪,不能排除图内激活误差.详见 [范围报告](results/interface_range_diagnostic.json).
- 第 34 层同一 16 个训练校准样本、同一 R1 的消融中,常规量化 RMSE=0.15042,Hessian=0.10360,减少 31.1%.支持权重量化误差是可修复的一部分,不能指定某个投影为唯一原因.
- 当前 43 分片每层使用 16 个训练校准样本的 Hessian,累计层误差优化关闭.先优化 10 分片,再复用并优化剩余 33 分片.SDK 量化分别约 3 分 56 秒、11 分 18 秒,合计 15 分 14 秒,不含初始化、导出、编译.复用已完成的 128 批校准,没有训练或微调原始 FP32 权重.
- 分片 A16 hidden 尺度不同,静态图使用官方 `Requantize` 显式转换,只接受同形状 INT16 单尺度接口,保留 Builder 检查.算子数值验证与实际双图导出已完成.
- SDK 临时对象可能为目录,兼容入口保留目录供审查,仅清理临时文件,避免双图成功导出后因目录清理异常误判失败.

修复已完成整模首轮实测.剩余 14.6 个百分点 Accuracy 差距仍未解决;需继续区分投影误差、图内激活误差及后段累积,不能用接近 1 的 TFLite/NPU 余弦宣称精度合格.

逐题分析中,FP32 正确而 NPU 错误 252 题,反向改善 106 题.不超过 128 Token 的 756 题只需一次 Prefill,FP32 正确 440 题、NPU 正确 323 题;因此剩余退化不能只归因于多块 Prefill 的 KV Cache 转移.长于 512 Token 的首轮题目仅 9 道,不据此推断长上下文精度.详见 [逐题误差统计](results/mmlu_error_analysis.json).

## 统一评价协议

MMLU 使用 [cais/mmlu](https://huggingface.co/datasets/cais/mmlu),固定 revision `c30699e8356da336a370243923dbaf21066bb9fe`,test SHA256 `74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7`.0-shot,seed=17,按 57 学科题数比例固定抽样 1000 题.不加 Chat Template 或额外特殊 Token,取空格加 A/B/C/D 的单 Token logits 评分.最长题目 968 Token,没有截断或丢题.FP32/NPU 的题目 ID、原始索引、学科、答案和 cases SHA 已逐项核对.板端千题评分约 10 分 49 秒.

WikiText2 使用 test 前 128 行原始文本,不加特殊 Token,每块 128 Token、重置上下文、评分 127 次下一 Token.两端原始文本 SHA、协议、58 块、7366 个评分位置与舍弃尾部 33 Token 对齐.板端评分约 36 秒.固定子集结果不冒充完整 PPL.

## 执行与产物

公共脚本统一位于仓库 `tools/llm/`.在 Ubuntu89 仓库调用 `bash models/gen_ai/llm/minicpm5_2b/deploy/run_native.sh STAGE`,阶段为 `prepare`、`calibrate`、`quantize`、`shape`、`compile`.

常规入口默认 `WEIGHT_OPTIMIZATION=none`,不默认开启长耗时优化.当前 v8 使用归档的 16 样本、分阶段 Hessian 实验脚本;直接运行默认入口不会生成同一 v8.量化记录须保存实际优化分片、样本数、累计层误差开关和旋转配置.测试集不参与校准.

量化后审核实际 TFLite 的 295 个投影权重为 INT4、接口为 INT16及完整 43 分片哈希.可用 `tools/llm/quantize_native.py --audit-only converter CONFIG -p asym4W_sym16A -tp asym4W_sym16A` 单独审核.打包读取成功记录中的实际前缀与优化配置;MMLU 使用 `1024c/prompt.dla` 并核对契约和清单.

独立 tokenizer 副本仅转换 BPE merges 序列化格式.校准配置使用单 EOS=130073,运行保留官方停止 Token 1/130073,默认 enable_thinking=false.

服务器模型包: `/tmp/hailongcodex/2026-10-14/minicpm5-w4-light-hessian-native-v8`.
板端模型包: `/root/hailong.he/open_models/minicpm5_w4_light_hessian_v8_native`.
首轮原始记录: `/tmp/hailongcodex/2026-10-14/minicpm5_light_hessian_first_round`.
工具归档目录: `/data/users/hailong.he/data/MTKG720/GAI_Toolkit/MiniCPM5-2B/`,明确标为项目适配脚本,不是 MTK 官方 MiniCPM5 专用 Toolkit.

最终 NAS 根目录采用用户指定的 G720 外部 `models/` 目录,模型包相对路径为 `gen_ai/llm/minicpm5_2b/minicpm5-2b/`.保留该模型目录已有的来源记录,复制后逐文件核对 SHA256,记录在模型包的 `delivery_verification.json` 中.服务器模型可运行,精度尚未通过质量验收.
