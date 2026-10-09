# MobileFaceNet 模型卡

| 项目 | 内容 |
| --- | --- |
| 任务 | 已对齐人脸的特征提取 |
| 模型论文 | https://arxiv.org/abs/1804.07573 |
| 第三方实现 | https://github.com/foamliu/MobileFaceNet |
| 固定提交 | `a687c71bea830e70d05fb3b38ddc7c68e1687e94` |
| 权重下载 | https://github.com/foamliu/MobileFaceNet/releases/download/v1.0/mobilefacenet.pt |
| 权重大小 | 4,135,271 字节 |
| 源码许可 | Apache-2.0，见 `LICENSE` |
| 输入 | 对齐人脸，RGB、112×112、NCHW FP32、ImageNet 均值及标准差归一化 |
| 输出 | 128 维特征向量；后续 L2 归一化及余弦相似度由应用实现 |
| 目标 | MT8189 / Genio 720 MDLA 5.3 |

正式测试使用对齐 LFW、6000 对及十折验证,示例复用其中三组验证对的实际板端预测.开放集拒识阈值、活体检测和端到端延迟仍未评估。
上游 v1.0 标签的网络结构与 v1.0 发布权重不匹配，因此源码固定到与权重参数布局对应的提交。

## 当前测试结果

LFW 全量 6000 对十折验证: 同协议 ONNX FP32 准确率为 99.38%,板端为 99.32%,下降 0.07 个百分点.独立 C++ 常驻模型平均 NPU 耗时为 0.52 ms,推理进程峰值 RSS 为 8.25 MiB.

结果见 [summary.json](results/summary.json),计时说明及三组实际验证效果见 [README](README.md#当前测试结果).
