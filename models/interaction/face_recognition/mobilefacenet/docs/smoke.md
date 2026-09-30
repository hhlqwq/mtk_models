# Genio 720 板端冒烟记录

## 运行信息

| 项目 | 结果 |
| --- | --- |
| 日期和运行编号 | 2026-09-24，`20260924T020219Z` |
| 模型源码 | `foamliu/MobileFaceNet` 提交 `a687c71bea830e70d05fb3b38ddc7c68e1687e94` |
| 权重 | v1.0 `mobilefacenet.pt` |
| 转换环境 | Ubuntu 89，`hhl_g720_8011`，MTK Converter 8.16.0 |
| 编译 | `ncc-tflite --arch=mdla5.3 --suppress-output --disallow-bridge` |
| 板端 | Genio 720 EVK，aarch64，Linux 6.6.137，Neuron Runtime 8.2.16 |
| 推理 | 三次 `neuronrt -m hw`，其中第 3 次重复第 1 次输入 |
| 结果 | `smoke_passed`，每次输出 128 个 INT8 值，重复输入输出完全一致 |

89 测试机的原始证据位于
`/data/users/hailong.he/github/mtk_models/models/interaction/face_recognition/mobilefacenet/examples/output/runs/20260924T020219Z/`。
板端运行目录为 `/root/hailong.he/open_models/mobilefacenet/demo/smoke/20260924T020219Z/`。
原始向量和对齐人脸图片均不进入 Git。

## 数值检查

- 固定权重严格加载成功，PyTorch 输出形状为 `(1, 128)`。
- 同一张对齐人脸的 PyTorch 与 ONNX 输出最大绝对差约 `5.13e-6`，余弦相似度约 `1.0000001`。
- ONNX 与板端反量化输出余弦相似度约 `0.9901`，最大绝对差约 `0.503`。
- 不同图片的两次板端输出余弦相似度约 `0.308`。这些图片未按独立身份评测，不能据此设定身份阈值。
- 三次输出均为有限非零值；第 1 次和第 3 次的板端原始输出完全一致。

本次只证明固定模型可在目标板硬件模式完成特征提取。正式人脸验证精度、误识率、
延迟、资源占用、活体检测和端到端机器人流程均未评测。
