# ViT-Base Patch16 224 模型卡

## 当前开源上游状态

正式开源上游实现、固定源码版本和官方权重尚未锁定.锁定前不得生成新的正式 MTK 产物,
也不得将下述历史 Qualcomm ONNX 结果计入当前交付状态.

候选上游必须由模型作者或官方开源项目发布,允许修改和交付,可从原始框架自行导出
ONNX,并能完成原始框架、ONNX、MTK NPU 三端精度对比.

## 历史 Qualcomm 衍生基线

- 交付形式参考：<https://huggingface.co/qualcomm/VIT>
- 历史实现：<https://github.com/qualcomm/ai-hub-models/tree/v0.61.0/src/qai_hub_models/models/vit>
- 历史模型：Qualcomm AI Hub Models v0.61.0 预导出 FP32 ONNX.
- 历史归档 SHA-256：`72b7d02dd5c3d1e09c59196ba14364549aac9e8ed1f8212ea5f5c78f7424632f`.

## 历史模型规格

- 参数量：86.6M.
- 输入：224×224 RGB.
- 输出：ImageNet-1K 1000 类 logits.
- 输入数值：NCHW RGB float32 `[0,1]`,mean/std 归一化位于 ONNX 图内.
- 历史来源页面标记的许可证：BSD-3-Clause；新上游许可证仍待锁定.

历史归档文件和解压后 ONNX 的 SHA-256 已记录.Qualcomm 设备性能仅作交付形式参考,
不能作为 Genio 720 实测结果.

## 历史交付产物

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
