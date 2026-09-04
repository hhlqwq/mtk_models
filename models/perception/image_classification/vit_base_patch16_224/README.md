# ViT-Base Patch16 224

## 模型信息

```text
模型: ViT-Base Patch16 224
任务: ImageNet-1K 图像分类
输入: 1×3×224×224 RGB
输出: 1×1000 classes
设备: MediaTek Genio 720 EVK
当前状态: 环境建设中
```

## 执行流程

```bash
cd /workspace/models/perception/image_classification/vit_base_patch16_224
./deploy/download_original.sh
./deploy/convert.sh
./deploy/build.sh
./deploy/deploy_board.sh
```

正式 Top-1 需要 ImageNet ILSVRC2012 验证集及标签映射。数据集缺失时只允许执行模型转换和
单图 Demo，不填写最终 Top-1。

## 交付状态

| 环节 | 状态 | 证据 |
| --- | --- | --- |
| Hugging Face 来源 | 已锁定 | `original/source_url.txt` |
| Qualcomm FP32 ONNX | 待下载 | `models/model_fp32.onnx` |
| MTK INT8 TFLite | 待执行 | `models/model_int8.tflite` |
| DLA | 待执行 | `models/model_int8.dla` |
| 板端 Demo | 待执行 | `examples/output/` |
| ImageNet Top-1 | 等待数据集 | `docs/accuracy.md` |
| 板端性能 | 待执行 | `docs/benchmark.md` |
