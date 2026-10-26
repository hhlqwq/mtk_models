# 输出误差统计

模型 `deploy/evaluate.sh` 在任务精度、性能和进程峰值 RSS 之外,同时计算 Cosine Similarity(CosSim)、RMSE 和 NRMSE.入口只启用统计,计算由 Python 完成,最终合并到唯一 `results/summary.json` 的 `output_comparison`.本次为代码接入,尚未运行参考生成、板端评价或数值测试;模型列表中的缺失值继续显示 `—`.

## 定义与展示

设同一输出头的浮点参考为 `r`,NPU 反量化或浮点输出为 `a`,将约定评价范围内该输出头的所有元素视为一个向量:

- `CosSim = sum(r * a) / sqrt(sum(r²)) / sqrt(sum(a²))`.
- `RMSE = sqrt(sum((a - r)²) / N)`.
- `NRMSE = 100 * sqrt(sum((a - r)²) / sum(r²))`,单位为百分比,等价于 `100 * RMSE / RMS(r)`.

使用 FP64 流式累加,不平均逐样本 CosSim 或逐样本 RMSE.多输出分别保存,不跨输出头混合不同量纲,也不生成跨头平均 RMSE.参考 RMS 为零时 NRMSE 留空;参考或实际输出为零向量时 CosSim 留空,并记录原因.形状不一致、空张量或非有限值直接报错.

模型总表的 G720、G5100 两列统一采用 `CosSim/RMSE/NRMSE/精度`,精度变化单独换行为 `(-1.28%)`.这里的 `-1.28%` 表示下降 1.28 个百分点,不是相对下降 1.28%.PPL 变化为绝对差值.多个任务保留 MMLU/PPL 等名称,多个输出头需要分别展示,不能把某个头的结果当作整模型误差.表格依据已完成的实际报告人工同步,缺失值不推算或填示例数字.

## 参考资源和执行

1. 在已有编译容器中执行本模型 `deploy/evaluate.sh`.普通模型使用已有浮点图计算任务参考,同时将同输入的后处理前浮点输出保存在一个 `output_reference.sqlite` 文件.参考文件和预测中间数据属于评价资源,不提交到代码仓库或放入 `results/`.
2. 将终端显示的参考文件与任务参考单独放入现有板端部署包.普通模型使用 `models/output_reference.sqlite`;MiniCPM 使用 `evaluation/mmlu1000/quality/output_reference.sqlite`,DeepSeek 使用 `evaluation/full/quality/output_reference.sqlite`.ResNet101 的已有参考上传步骤同时上传此文件.新增代码需要随部署包同步,不能只上传参考文件后运行旧版板端程序.
3. 用户登录板端执行 `bash evaluate.sh`.普通模型在原有 NPU 推理中比较输出,SQLite 只读打开,按量化前输入的逻辑形状和 FP32 内容配对,不依赖样本遍历顺序.重复输入也核对出现次数,最终覆盖必须与参考一致.输出名称不同仅在形状唯一时允许匹配,多头形状相同不能猜测.
4. 临时目录只保存一份 `work/output_metrics.json` 汇总,供正式报告发布时读取;不保存 NPU 张量副本或逐样本指标.浮点参考库只在成功结束时标记完整,损坏、未完成、范围不全或输入不匹配的参考直接报错.普通模型同时核对参考所关联的已编译 DLA;LLM 沿用既有模型配置、Tokenizer、Token 和图契约核对.

缺少参考时仍可记录原有任务精度,`output_comparison.status` 为 `unavailable`,附具体原因;不得沿用旧运行的输出误差.首次准备这些参考会增加浮点输出的存储和传输量,必须按实际完整评价范围准备.已有的标量任务精度报告不能恢复完整输出,需要在用户实际执行评价时重新生成参考,不重新编译模型.

普通模型的纯 NPU 耗时仍只测 SDK 调用,比较、参考读取和统计均在该计时范围外.推理进程峰值 RSS 会包含新增比较逻辑及其缓冲区,因此不能直接与不含该逻辑的历史 RSS 当作同实现比较.YAMNet 的窗口统计与其他模型一样纳入实际评价进程;报告保留具体计时和内存范围.

## 已接入范围

| 模型 | 输出比较范围 |
| --- | --- |
| YOLOv5s、YOLOv8n、FastSAM | 完整 COCO val2017 输入,逐原始输出头,在解码、NMS 或掩码处理前比较 |
| ResNet101、ViT-B/16 | 完整 ImageNet val2012 的分类 logits,不使用 Top-5 类别索引计算误差 |
| RTMPose WholeBody | 完整 104125 个人体裁剪,分别统计两个 SimCC 输出 |
| MobileFaceNet | LFW 协议涉及的全部对齐人脸,在特征归一化和阈值评分前比较 |
| Depth Anything V2 Small | DA-2K 协议涉及的全部图片,在恢复原图分辨率前比较深度输出 |
| YAMNet | 全部约定音频的每个 96 帧窗口,在窗口平均和标签投影前比较 |
| Whisper Tiny | 全部 test-clean 音频的 Encoder 输出,使用同一 Log-Mel 实现及 FP16 输入边界;不包含会因 Token 和 Cache 分叉而失去逐元素对应的自回归 Decoder |
| MiniCPM5、DeepSeek | WikiText2 约定全部文本块,每块前 127 个位置的完整词表 Prefill logits;真实 Token teacher forcing,每块重置上下文;不包含 MMLU 或 Decode |
| MIT BEVFusion | 全部 6019 帧,性能/RSS 采集后独立重放,逐 NPU 子图对比同输入的已有 FP32 ONNX,不包含 CPU 子图、上游累计误差或最终检测框 |

YOLOv5s 浮点参考的缩放取整与现有板端实现对齐为 FP32 比例和正数四舍五入.Whisper 浮点参考改为复用板端 Log-Mel 实现,包括 FP16 输入边界.这两项要求重新生成匹配参考,不能将新误差与旧协议任务精度拼成同次报告.

BEVFusion 已有 ONNX 随部署包提供,直接在独立重放阶段生成同输入子图参考,无需 SQLite 参考库.该阶段会额外执行完整 NPU 重放与子图 CPU 浮点推理,不纳入先前采集的性能和 RSS,阶段进度单独显示.

## 新模型接入

主机复用 `tools.output_metrics.record_reference(source, inputs, outputs)` 和 `finish_reference()`,前者接收源模型路径、按契约顺序排列的量化前输入列表及具名浮点输出字典,后者仅在完整评价成功后调用.使用 `tools.deploy.context` 时参考目录自动来自本模型配置;独立适配器应设置 `MTK_OUTPUT_REFERENCE` 和 `MTK_OUTPUT_MODEL_DIRS`,分别指定一个参考库文件及已有 DLA 所在目录的 JSON 列表.不能在编译阶段保存正式评价统计.

板端在正式推理开始前创建 `OutputComparison(ctx)`,每次正式 NPU 调用后使用量化前输入和反量化输出调用 `update()`,预热和示例不更新;结束时调用 `finish()`.量化前输入和浮点参考必须采用同一预处理实现,不能改用反量化后的量化输入进行配对.多输出字典优先使用源图语义名称,并保留准确逻辑形状.

打包时包含 `tools/output_metrics.py` 及其 `tools/files.py`、`tools/evaluation.py` 依赖.复用 `tools.summary.publish` 即可合并本次统计;自行发布的模型通过 `report_metrics(ctx)` 读取.输出比较范围必须如实标注,不能因部分张量可以比较就宣称全模型全部输出已验证.G5100 无板端实测时继续保留缺失状态,不能复制 G720 指标.
