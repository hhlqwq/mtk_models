# MobileFaceNet 精度

主指标为 LFW 6000 对、10 折验证准确率。正确协议要求先做五点人脸对齐，再进行 RGB 与 ImageNet 均值、标准差归一化。

**正式准确率待重测。** 历史运行使用未对齐人脸和错误的输入归一化，原始[板端报告](../results/full_accuracy/20260928_mobilefacenet_lfw_full_v1/summary.json)与[参考报告](../results/reference_accuracy/lfw_fp32_v1/summary.json)仅供定位问题，不作为正式精度。开放集误识率和现场识别率也尚未评测。
