# FastSAM-s 精度

COCO val2017 全量 5000 张图片，类别无关实例分割。三端采用相同的 640×640 输入、置信度 0.001、NMS IoU 0.9、每图最多 100 个实例；主指标为 segm AR@100。

| 后端 | segm AR@100 |
| --- | ---: |
| 官方 PyTorch FP32 | 0.391 |
| FP32 ONNX | 0.390 |
| 独立校准 INT8 DLA | 0.376384 |

板端较 ONNX 低约 0.014。结果见[PyTorch](../results/reference_accuracy/coco_pytorch_v2/summary.json)、[ONNX](../results/reference_accuracy/coco_fp32_v2/summary.json)和[板端报告](../results/full_accuracy/20260929_fastsam_imagenet100_full_v2/summary.json)。旧 DLA 的校准图片与评测集重叠，且旧参考端与板端置信度阈值不同，因此旧结果不作为同协议对比。
