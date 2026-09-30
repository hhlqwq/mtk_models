# RTMPose Body2d 精度

使用 OpenMMLab MMPose v1.3.2 官方 RTMPose-M、COCO-WholeBody 验证集和相同的 104125 个人体检测框。主指标为 WholeBody AP。

| 后端 | WholeBody AP |
| --- | ---: |
| PyTorch FP32 | 0.5702 |
| ONNX FP32 | 0.5703 |
| MTK NPU INT8 | 0.5324 |

板端较 ONNX 下降 0.0380，即 3.80 个百分点。该数值只描述本项目同协议后端差异。机器可读结果见 [三后端报告](accuracy_comparison_20260914.json)。
