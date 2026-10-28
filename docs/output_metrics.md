# 小样本输出数值对照

CosSim、RMSE、NRMSE 仅在 compile.sh 的固定小样本检查中计算,全量 evaluate.sh 不收集浮点张量、不运行输出对照.任务精度仍按同协议 Top-1、mAP、WER 等独立评价.

## 范围与文件

普通模型从现有 compile 固定子集中选择最多 5 次输入,不会减少原有通路检查样本数量.LLM 对照示例首个 Prefill 块的有效位置,不包含自回归 Decode;Whisper 对照 Encoder,不包含 Decoder;BEVFusion 对照单帧的 NPU 子图,不混合 CPU 子图误差.终端明确显示实际范围、输入次数和各输出结果,同时写入 summary_test.json,不混入全量 summary.json.

主机参考保存在本地 `/tmp/hailongcodex/当天日期/<模型>/compile/output_reference.npz`,随后上传板端 `build/reference/output_reference.npz`,每次 compile 覆盖.使用普通数组文件,不依赖数据库,不向 NAS 写评价张量.文件记录输入签名、源图及对应 DLA 版本,板端使用相同预处理输入匹配.缺参考、版本或覆盖不符时明确提示,不借用历史指标.

## 定义

- CosSim: 对应输出全部元素的点积除以两侧向量范数乘积.
- RMSE: 对应输出全部元素的均方根误差.
- NRMSE: `100 * RMSE / RMS(reference)`,单位为百分比.

逐输出累加 FP64 充分统计量,不跨不同输出头平均.零向量或零参考能量造成指标未定义时明确标记.参考与硬件结果必须属于同模型版本、同输入、同形状和同输出语义.

### compile 子集精度检查

compile.sh 在主机计算固定子集的浮点任务精度,参考及指标保存在主机本地日期过程目录,随当前模型自动上传;板端推理相同子集后,将浮点精度、NPU 精度和精度损失写入部署目录 results/summary_test.json.板端总 compile 完成后回传子集摘要和示例,服务器收到的是包含两端对比的子集摘要.全量 evaluate.sh 仍只更新 results/summary.json,两份报告互相保留.

| 任务 | compile 子集 | 抽样约束 |
| --- | --- | --- |
| ResNet101、ViT 分类 | 1000 张 | ImageNet val,保留前 5 张数值/展示输入 |
| YOLOv5、YOLOv8 检测 | 300 张 | COCO val2017,保留展示图片 |
| FastSAM 分割 | 300 张 | 保持本项目 AR@100 协议 |
| RTMPose | 至少 500 个人体实例 | 按完整图片分组,不截断一张图的人体框 |
| MobileFaceNet | 500 对 | 十折,每折各 25 对同人和异人 |
| Depth Anything V2 | 100 张 | 保持 DA-2K 点对协议 |
| YAMNet | 250 条音频 | ESC-50 每类每折各 1 条 |
| Whisper | 50 条音频 | LibriSpeech test-clean,保留展示音频 |
| MiniCPM、DeepSeek | MMLU 100 题,WikiText2 16 块 | 覆盖 57 学科,seed=17;每块 128 Token |
| BEVFusion | 最多 100 帧 | 当前 mini_val 为全部 81 帧,如实显示范围 |

间隔抽样参考高通默认方式:先对样本标识排序,保留必要示例后,对剩余 N 个候选和 K 个名额设置 step=max(1,N//K),选择下标 0、step、2×step 等,最后按标识排序.因此可复现且跨完整划分取样,不会只取前 K 个样本.高通和本项目的必需样本、数据集及预后处理可能不同,不能直接比较其发布精度.MMLU 使用固定种子的按学科比例分层抽样;ESC-50 保留类别与折覆盖.

精度使用 0-1 比例.损失的正值统一表示变差:Top-1、AP、AR 等为 100×(浮点−NPU),WER 为 100×(NPU−浮点),单位为百分点;PPL 保存 NPU−浮点的绝对增量和相对增幅.主机参考属于评价输入,最终 summary_test.json 由板端成功比较后发布.不得以两个不同子集或不同 DLA 计算损失.

CosSim、RMSE、NRMSE 仍只比较最多 5 次输入,不保存整个子集的浮点输出.主机过程数据保存在本地 compile 临时目录,必要参考上传板端 build/reference/;板端中间文件复用本模型 build/work/.子集不采集或发布性能和 RSS,这些指标仍属于全量 evaluate.sh.

所有参考准备及上传已包含在 compile.sh 中,无需额外 Python 命令.全量 evaluate.sh 自动上传已有模型的全量浮点评价结果和必要输入.两端精度通过 tools.evaluation.accuracy_result 生成统一接口,公共汇总只读取该接口,不猜测模型专用字段.

CosSim、RMSE、NRMSE 在终端打印,并按输出名称与比较范围写入 summary_test.json 的 output_errors.普通模型及 LLM/Whisper 使用主机生成的参考;BEVFusion 保留单帧的同子图同输入对比,使用板端 CPU ONNX 对照各 NPU 子图,在报告中明确 reference_backend,不把 CPU 子图混入误差.零向量导致的未定义指标保存 null 和原因.

YAMNet 子集处理 250 条音频,主指标实际评价 folds 2-5 中有类别映射的 188 条;全量处理 2000 条,主指标评价 1504 条.统一精度接口的样本数与标识对应实际主指标范围,不能把处理数量当作评分数量.

ViT 的数值参考使用板端实际预处理后的前 5 张输入,主机只执行对应 ONNX,两端核对原始图片一致性.这样避免 OpenCV 在 x86 和 ARM 上的缩放差异导致输入指纹不匹配;不跳过输入或 DLA 的一致性检查.
