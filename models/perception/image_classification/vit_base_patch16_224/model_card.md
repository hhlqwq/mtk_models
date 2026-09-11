# ViT-Base Patch16 224 模型卡

## 正式开源上游

- 官方项目：<https://github.com/pytorch/vision>
- 固定版本：`v0.15.1`,与项目既有 Torch 2.0.0 / TorchVision 0.15.1 环境一致.
- 模型构造器：`torchvision.models.vit_b_16`.
- 权重枚举：`ViT_B_16_Weights.IMAGENET1K_V1`.
- 官方权重：<https://download.pytorch.org/models/vit_b_16-c867db91.pth>
- 文件大小：346,328,529 bytes.
- SHA-256：`c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9`.
- 许可证：TorchVision BSD-3-Clause.

`deploy/export_onnx.py` 从本地官方 `.pth` 自行生成精确 GELU 基线 ONNX 和 tanh GELU
MTK 兼容候选 ONNX,不使用任何第三方预导出模型.

## 历史 Qualcomm 衍生基线

- 交付形式参考：<https://huggingface.co/qualcomm/VIT>
- 历史实现：<https://github.com/qualcomm/ai-hub-models/tree/v0.61.0/src/qai_hub_models/models/vit>
- 历史模型：Qualcomm AI Hub Models v0.61.0 预导出 FP32 ONNX.
- 历史归档 SHA-256：`72b7d02dd5c3d1e09c59196ba14364549aac9e8ed1f8212ea5f5c78f7424632f`.

## 正式模型规格

- 输入：224×224 RGB.
- 输出：ImageNet-1K 1000 类 logits.
- 输入数值：NCHW RGB float32 `[0,1]`,TorchVision ImageNet mean/std 归一化位于导出图内.
- 官方基准：ImageNet-1K Top-1 81.072%、Top-5 95.318%.
- 参数量：86,567,656.

历史归档文件和解压后 ONNX 的 SHA-256 已记录.Qualcomm 设备性能仅作交付形式参考,
不能作为 Genio 720 实测结果.

## 正式 Genio 720 交付结果

| 产物 | 大小 (bytes) | SHA-256 |
| --- | ---: | --- |
| `model_fp32.onnx` | 346,443,880 | `5cf7159b106ef651b0800c0b75b04629ebf173c44e18b9a018a4e38d1d89ce7c` |
| `model_mtk_compatible.onnx` | 346,458,248 | `df15d9dad8e9952a865ff64e431e24baa4f64884d2006a1a0e4d7db3a0899692` |
| `model_int8.tflite` | 90,450,160 | `74976e225c06ffb1e82d878360e70ad267eba17b819a4e793ff5ced0afa330c1` |
| `model_int8.dla` | 88,003,676 | `2aec455949fa25420611645eb97617433046b6d488d87e66edf26e8246b85d67` |

ImageNet val 50,000 张正式结果：FP32 ONNX Top-1/Top-5 为 80.64%/95.10%,
Genio 720 NPU INT8 为 79.38%/94.69%.纯 NPU 平均延迟为 51.7129 ms,
完整结果见 `docs/imagenet_accuracy_20260911.json`.

## 历史 Qualcomm 交付产物

| 产物 | 大小 (bytes) | SHA-256 |
| --- | ---: | --- |
| `model_fp32.onnx` | 347,444,621 | `e611b7797dcd835af914e69e732c931c3fab6102e781ea61afcc1a6bf1f86138` |
| `model_mtk_compatible.onnx` | 347,417,235 | `dcae9ec010cc61b794da0299826f986cd0f34531cf4275fbc23461e2fd23c6e8` |
| `model_int8.tflite` | 90,430,376 | `829f15a1891604e1fdf632ce6bf4d8f55abd0bc8d07942ee4169e5240d82aca1` |
| `model_int8.dla` | 87,979,244 | `f0b14052868d571d4a5fce13935802815bd4ebb5e0d7c9a97c7eddb4d0d1611a` |

## 历史 Genio 720 验证结果

以下结果只适用于 Qualcomm v0.61.0 预导出 ONNX 的衍生模型,不得作为待选开源上游
ViT 的正式结果.ILSVRC2012 val 全部 50,000 张图片的 FP32 ONNX Top-1/Top-5 为
80.64%/95.11%, MTK NPU INT8 Top-1/Top-5 为 79.40%/94.64%. 排除 100 张
PTQ 校准图片后的 49,900 张独立结果为 FP32 80.63%/95.10%、NPU INT8
79.38%/94.63%. 完整运行 ID、样本完整性和证据哈希见
[`docs/accuracy.md`](docs/accuracy.md) 与
[`docs/imagenet_accuracy_20260908.json`](docs/imagenet_accuracy_20260908.json).
