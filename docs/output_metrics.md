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

### compile 子集精度检查

compile.sh 在固定验证子集上计算任务精度,终端显示同一子集的浮点精度、NPU 精度及变化,不改写 results/summary.json.图像任务默认 100 张;LFW 保留十折共 100 对,ESC-50 保留 50 类和 5 折共 250 条音频,Whisper 使用固定 5 条音频,LLM 使用配置指定的 MMLU 小题集与 WikiText2 少量块,BEVFusion 最多使用 100 帧,划分不足 100 帧时使用全部帧并明确显示实际数量.子集成绩只用于快速检查,不能代替全量成绩.

任务精度与数值误差分开计算: CosSim、RMSE、NRMSE 仍只比较最多 5 次输入,不保存整个子集的浮点输出.主机参考保存在本地 compile 临时目录,上传板端 board/;预测及评分中间数据复用本模型 work/.正式 evaluate.sh 才更新 results/summary.json 并统计性能与 RSS.

已有 ONNX/DLA 只需更新子集参考时,在已有编译容器内运行:

```bash
python tools/compile_accuracy.py models/deploy
```

该命令按模型准备并上传子集任务精度和少量数值参考,不重新导出、量化、编译或执行 NPU 推理.随后登录板端执行 `bash deploy/compile.sh`.正常从头编译时,原有 `models/deploy/compile.sh` 已包含这些参考准备步骤.

ViT 的数值参考使用板端实际预处理后的前 5 张输入,主机只执行对应 ONNX,两端核对原始图片一致性.这样避免 OpenCV 在 x86 和 ARM 上的缩放差异导致输入指纹不匹配;不跳过输入或 DLA 的一致性检查.
