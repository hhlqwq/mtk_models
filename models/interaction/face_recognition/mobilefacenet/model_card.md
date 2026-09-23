# MobileFaceNet 模型卡

| 项目 | 内容 |
| --- | --- |
| 任务 | 已对齐人脸的特征提取 |
| 模型论文 | https://arxiv.org/abs/1804.07573 |
| 第三方实现 | https://github.com/foamliu/MobileFaceNet |
| 固定提交 | `a687c71bea830e70d05fb3b38ddc7c68e1687e94` |
| 权重下载 | https://github.com/foamliu/MobileFaceNet/releases/download/v1.0/mobilefacenet.pt |
| 权重大小 | 4,135,271 字节 |
| 权重 SHA-256 | `90a00ba1d8b0b688af3deb731ed53dca582e6106805d1bc3cfdef55f570493f4` |
| 源码许可 | Apache-2.0，见 `LICENSE` |
| 输入 | BGR、112×112、NCHW FP32、`(x - 127.5) / 128` |
| 输出 | 128 维特征向量；后续 L2 归一化及余弦相似度由应用实现 |
| 目标 | MT8189 / Genio 720 MDLA 5.3 |

权重和训练数据是否满足产品用途仍待单独确认。上游示例图片仅用于转换校准及板端冒烟，
不提交原图、人脸特征或身份信息。精度、开放集拒识阈值、活体检测和端到端延迟均未评估。
上游 v1.0 标签的网络结构与 v1.0 发布权重不匹配，因此源码固定到与权重参数布局对应的提交。
