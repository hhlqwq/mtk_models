# YOLOv5s 精度报告

状态：尚未评测，不得引用为最终结果。

| 后端 | COCO mAP@0.5 | COCO mAP@0.5:0.95 | 验证集 |
| --- | ---: | ---: | --- |
| 原始 PyTorch | 待测 | 待测 | COCO val2017 |
| FP32 ONNX | 待测 | 待测 | COCO val2017 |
| MTK NPU INT8 | 待测 | 待测 | COCO val2017 |

## 验收规则

- 三种后端必须使用同一份 COCO val2017 图片、标注、预处理和后处理参数。
- 报告 ONNX 相对 PyTorch，以及 MTK NPU INT8 相对 PyTorch/ONNX 的绝对 mAP 损失。
- 少量图片的结果对齐仅记为冒烟测试，不填写正式 mAP。
- 正式结果必须记录评测命令、数据集路径、样本数和产物 SHA-256。
