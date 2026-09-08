# ViT-Base Patch16 224 模型卡

## 来源

- Hugging Face：<https://huggingface.co/qualcomm/VIT>
- Qualcomm AI Hub Models：<https://github.com/qualcomm/ai-hub-models/tree/main/src/qai_hub_models/models/vit>
- 模型版本：Qualcomm AI Hub Models v0.61.0 预导出 FP32 ONNX。
- 原始归档 SHA-256：`72b7d02dd5c3d1e09c59196ba14364549aac9e8ed1f8212ea5f5c78f7424632f`。

## 模型规格

- 参数量：86.6M。
- 输入：224×224 RGB。
- 输出：ImageNet-1K 1000 类 logits。
- 输入数值：NCHW RGB float32 `[0,1]`，mean/std 归一化位于 ONNX 图内。
- 来源页面许可证：BSD-3-Clause。

下载后必须记录归档文件和解压后 ONNX 的 SHA-256。Qualcomm 设备性能仅作对标，不能作为
Genio 720 实测结果。
