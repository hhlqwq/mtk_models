# 小样本输出数值对照

CosSim、RMSE、NRMSE 仅在 compile.sh 的固定小样本检查中计算,全量 evaluate.sh 不收集浮点张量、不运行输出对照.任务精度仍按同协议 Top-1、mAP、WER 等独立评价.

## 范围与文件

普通模型从现有 compile 固定子集中选择最多 5 次输入,不会减少原有通路检查样本数量.LLM 对照示例首个 Prefill 块的有效位置,不包含自回归 Decode;Whisper 对照 Encoder,不包含 Decoder;BEVFusion 对照单帧的 NPU 子图,不混合 CPU 子图误差.终端明确显示实际范围、输入次数和各输出结果,不把小样本统计写入全量 summary.json.

主机参考保存在本地 `/tmp/hailongcodex/当天日期/<模型>/compile/output_reference.npz`,随后上传板端 `board/output_reference.npz`,每次 compile 覆盖.使用普通数组文件,不依赖数据库,不向 NAS 写评价张量.文件记录输入签名、源图及对应 DLA 版本,板端使用相同预处理输入匹配.缺参考、版本或覆盖不符时明确提示,不借用历史指标.

## 定义

- CosSim: 对应输出全部元素的点积除以两侧向量范数乘积.
- RMSE: 对应输出全部元素的均方根误差.
- NRMSE: `100 * RMSE / RMS(reference)`,单位为百分比.

逐输出累加 FP64 充分统计量,不跨不同输出头平均.零向量或零参考能量造成指标未定义时明确标记.参考与硬件结果必须属于同模型版本、同输入、同形状和同输出语义.
