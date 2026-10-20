# MiniCPM5-2B

编译与转换步骤统一在 Docker 容器内执行: 先在仓库根目录运行 `bash docker/enter_container.sh`,再进入本模型目录运行原有 `deploy/run.sh` 或 `deploy/run_native.sh` 入口.模型脚本不再调用 Docker 或在主机和容器之间复制文件; 上传完成后,板端入口仍在开发板执行.

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
7. **继续核对当前交付模型并筛选局部修复.** 重新完成连续 43 分片诊断,重点复核后段误差.保持全 W4A16,串行验证最后一层的权重量化范围优化,以及将 Hessian 校准从 16 个样本扩至包含原样本的 32 个样本.两项处理均改善单层平均 RMSE,但同输入四块整模预检的 PPL 变差,因此均不采用.随后发现这两项实验只核对了 R1,逐层 R2 不一致;候选不采用的结论保留,但不能把变化单独归因于优化方法或样本数量.详细配置和限制见 [优化过程记录](results/optimization_diagnostic.json).后续控制实验同时固定实际 R1/R2.整模预检一次只加载一个分片,避免同时缓存全模型分片导致显存不足.
8. **隔离权重误差并修复激活截断.** 使用实际 INT4 权重反量化,与 FP32 运算组合,分离权重误差和量化图误差.检查最后一层内部 MLP 乘积、down 投影和残差输出,确认存在少量但幅度较大的 A16 截断.保持全部 W4 常量及外部接口不变,为相关 A16 域增加 12.5% 余量并同步零偏置尺度,单层最大误差明显下降.等价的 SwiGLU up/down 通道均衡也有小范围改善,但未达到可替换交付模型的验证要求.因果控制、具体数值与候选验证状态见 [根因与修复记录](results/root_cause_diagnostic.json).

过程记录保留每一步的配置、依据和处理结论,当前结果表不保留实验编号或历史候选成绩.

## 当前精度分析

| 当前模型平均余弦 | Prefill,508 个评分位置 | Decode,64 个位置 |
|---|---:|---:|
| FP32 -> TFLite | 0.97083 | 0.97338 |
| FP32 -> NPU | 0.97115 | 0.97385 |
| TFLite -> NPU | 0.99955 | 0.99971 |

Prefill 完整 512 个输出位置的 FP32 -> NPU 平均余弦为 0.97105.四块诊断用于定位误差,不替代 MMLU Accuracy 与 WikiText PPL.主要损失已在量化阶段产生,板端新增误差较小.

上述定长诊断不代表所有变长输入的软件和硬件输出一致.额外选择覆盖 41~968 Token、1~8 块 Prefill 的 16 个 MMLU 问题,TFLite/NPU 有 14 题最终选择一致、2 题预测不同.板端重跑这 16 题,分数与原千题记录完全相同,没有观察到该组输入的重复运行漂移.两个排序差异分别出现在单块和多块输入,仍需继续分析编译与软件执行的数值差异.此检查不改写首轮千题 Accuracy,详见 [变长执行诊断](results/mmlu_execution_diagnostic.json).

当前交付模型已完成整模首轮实测.剩余 14.6 个百分点 Accuracy 差距仍未解决,不能用接近 1 的 TFLite/NPU 余弦宣称精度合格.后续局部修复的验证状态单独记录,不冒充当前交付成绩.

当前交付模型的连续 43 分片诊断已于 2026-10-15 重新核对.单块 128 Token 输入上,第 41 层自身相对 RMSE 为 3.84%,接收前面量化层输出后为 17.07%;LM Head 自身为 4.62%,累计 logits 为 19.85%.这进一步表明只修复 LM Head 不足以解决后段累积.该诊断在主机上定位量化误差,不替代板端 Accuracy/PPL.主 A16 接口四块约 1.56 亿个数值中 73 个超范围,包含相邻接口重复统计;低占比不能证明截断影响可以忽略.

同一实际旋转基下,第 41 层只保留 W4 权重、恢复 FP32 运算后 RMSE 仍为 0.24495,完整 W4A16 为 0.24999.单独量化各投影的控制实验中,MLP down/up 影响最大;down 权重相对 L2 误差约 25.0%,每输出通道一个尺度覆盖 6144 个输入权重.这确认该层存在明显的权重近似损失,尚不能把这个比例推广为整模 Accuracy 掉点的精确归因.现有公开转换接口的粒度限制也不等于已证明硬件不支持更细分组.

另外确认了可修复的激活截断: 一个参考输出为 -157.483,原量化输出恰好停在 INT16 下界对应的 -152.048;MLP 乘积和 down 输出也有越界.扩展相关 A16 域、同步零偏置尺度后,保持 W4 权重不变,该层最大误差从 5.435 降至 2.869,RMSE 降至 0.24516,已接近该控制下的权重误差.

该修复已进一步导出两张静态图,逐张核对 295 个 INT4 常量和外部接口保持不变,编译 Prefill DLA 并完成真实板端首轮评价.全模型范围扩展使 WikiText PPL 仅改善约 0.11%,而 MMLU 净少答对 5 题;其中 59 题由错变对、64 题由对变错.因此不采用这个整体扩展方案,不继续编译其 Decode DLA,当前双 DLA 交付包保持不变.扩大范围也会使量化步长变粗,局部截断减少并不保证整模选项排序更准确.这次筛选复用了固定首轮子集,属于开发诊断,不能当作独立留出集验证.

本轮未能完全修复主要精度损失: 调整 A16 范围不能恢复已经被近似的 W4 权重,几项低成本权重修复也未获得足以替换交付模型的整模证据.更细的 W4 分组、面向 MLP 的联合通道优化及更有针对性的范围调整尚未在当前双 DLA 链路验证;不能把公开转换接口中没有观察到分组配置等同于硬件不支持,也不能宣称 W4A16 已到优化极限.详细控制、限制和本轮全部原始脚本/结果的归档哈希见 [根因与修复记录](results/root_cause_diagnostic.json).

变长执行排查中,对一题分别修改未来 padding、被掩码的 KV 和掩码负值,板端完整 logits 均保持逐值相同;软件 FP16 模拟也未消除先前两题的排序差异.这些控制未支持“该题的掩码泄漏”解释,但不能证明所有输入都不存在执行误差.两个排序差异题均不是 FP32 正确而 NPU 错误的掉点题,不足以解释首轮主要 Accuracy 损失.

逐题分析中,FP32 正确而 NPU 错误 252 题,反向改善 106 题.不超过 128 Token 的 756 题只需一次 Prefill,FP32 正确 440 题、NPU 正确 323 题;因此剩余退化不能只归因于多块 Prefill 的 KV Cache 转移.长于 512 Token 的首轮题目仅 9 道,不据此推断长上下文精度.详见 [逐题误差统计](results/mmlu_error_analysis.json).

## 当前短生成功能检查

使用官方 Chat Template、关闭思考模式,选择 4 个简单算术和常识问题,每题分别执行文本输入与 Token 输入,每次最多生成 64 Token.8 次真实板端运行均完成;4 题都含正确答案并正常遇到停止 Token,两种输入的回答全部相同,未出现重复输出.该范围不能代表长对话稳定性.

按“只输出所要求答案”的严格格式检查为 3/4: 问题要求只输出 `3 + 4` 的结果,官方 FP32 输出 `7`,当前 NPU 输出 `3 + 4 = 7`.其余 3 题与同输入官方 FP32 的生成 Token 完全相同.因此不能把“答案正确、能停止”当作指令遵从全部通过.原始 Token、回答、停止状态与同题 FP32 参考统一保存在 [短生成记录](results/native_smoke.json),不按中英文拆分结果或性能.

## 统一评价协议

MMLU 使用 [cais/mmlu](https://huggingface.co/datasets/cais/mmlu),固定 revision `c30699e8356da336a370243923dbaf21066bb9fe`,test SHA256 `74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7`.0-shot,seed=17,按 57 学科题数比例固定抽样 1000 题.不加 Chat Template 或额外特殊 Token,取空格加 A/B/C/D 的单 Token logits 评分.最长题目 968 Token,没有截断或丢题.FP32/NPU 的题目 ID、原始索引、学科、答案和 cases SHA 已逐项核对.板端千题评分约 10 分 49 秒.

WikiText2 使用 test 前 128 行原始文本,不加特殊 Token,每块 128 Token、重置上下文、评分 127 次下一 Token.两端原始文本 SHA、协议、58 块、7366 个评分位置与舍弃尾部 33 Token 对齐.板端评分约 36 秒.固定子集结果不冒充完整 PPL.

## 执行与产物

公共脚本统一位于仓库 `tools/llm/`.在编译容器的项目根目录调用 `bash models/gen_ai/llm/minicpm5_2b/deploy/run_native.sh STAGE`,阶段为 `prepare`、`calibrate`、`quantize`、`shape`、`compile`.

常规入口默认 `WEIGHT_OPTIMIZATION=none`,不默认开启长耗时优化.当前交付模型使用归档的 16 样本、分阶段 Hessian 配方;直接运行默认入口不会生成相同模型.量化记录须保存实际优化分片、样本数、累计层误差开关和旋转配置.测试集不参与校准.

量化后审核实际 TFLite 的 295 个投影权重为 INT4、接口为 INT16及完整 43 分片哈希.可用 `tools/llm/quantize_native.py --audit-only converter CONFIG -p asym4W_sym16A -tp asym4W_sym16A` 单独审核.打包读取成功记录中的实际前缀与优化配置;MMLU 使用 `1024c/prompt.dla` 并核对契约和清单.

独立 tokenizer 副本仅转换 BPE merges 序列化格式.校准配置使用单 EOS=130073,运行保留官方停止 Token 1/130073,默认 enable_thinking=false.

项目适配工具的归档位置由本地记录维护,这些脚本不是 MTK 官方 MiniCPM5 专用 Toolkit.

最终 NAS 根目录采用用户指定的 G720 外部 `models/` 目录,模型包相对路径为 `gen_ai/llm/minicpm5_2b/minicpm5-2b/`.保留该模型目录已有的来源记录,复制后逐文件核对 SHA256,记录在模型包的 `delivery_verification.json` 中.服务器模型可运行,精度尚未通过质量验收.

## 板端目录管理

板端统一入口为 `/root/hailong.he/open_models/minicpm5_2b/`.当前交付包放在 `current/`,固定首轮评价数据和结果放在 `results/first_round/`,当前模型的数值诊断放在 `results/validation/`.进入 `current/` 后执行 `bash scripts/run.sh`;同一模型重复测试使用不同运行编号,结果写入 `current/results/`,不复制整个模型包.

只有权重、静态图或量化接口变化时才建立独立 `candidate/`,候选未通过对应评价前不得覆盖 `current/`.没有待验证候选时不创建空候选目录.已淘汰模型及历史诊断只保留配置、评价范围、精度结论和必要哈希记录,不长期保留模型包及中间数据.先生成文档记录再清理历史文件,当前交付包及对应评价资料保留.

2026-10-20 目录整理保留当前全 W4A16、16 样本 Hessian 双 DLA,不改模型、量化配置或既有评价成绩.历史报告中的绝对路径是当时的运行记录,不重写;现行路径与归档位置以板端本目录的 `README.md` 和 `layout_migration.json` 为准.本次只核对迁移完整性,不重新运行推理或评价.

随后按用户要求删除了 89 上 8 个历史目录中的 132 个文件,合计 11.07 GB.实验摘要保存在原归档记录目录的 `historical_experiments.md` 和 `historical_experiments.json`,原文件哈希保存在 `inventory_before.json`.这些记录不包含可恢复的历史模型包;诊断小子集指标不与首轮千题或完整首轮 PPL 直接比较.

## 模型资源路径

通过不提交 Git 的 `configs/storage.local.json` 读取 `models_root` 和 `dataset_root`.本模型资源根目录为 `models_root/gen_ai/llm/minicpm5_2b/`: `original/` 保存官方模型,`native/` 保存当前转换、量化和校准资源,`compiled/` 保存编译包归档.原生入口自动读取这些配置,实际完整路径仅保存本地.现有交付包路径按上文记录使用.仓库中的 `models/` 只保留源码、文档和公开来源记录.


模型与原生转换参数读取根目录的 `config.yaml`,公共工具链环境在进入 Docker 时自动加载.`deploy/run_native.sh` 保留原阶段名称,各阶段对应独立 Python 文件.
