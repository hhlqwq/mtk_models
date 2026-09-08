# RTMPose Body2d 精度报告

状态：正式评测数据和人体框来源尚未确认.

| 后端 | WholeBody AP | WholeBody AR | 验证配置 |
| --- | ---: | ---: | --- |
| 来源 PyTorch | 待测 | 待测 | 待确认 |
| FP32 ONNX | 待测 | 待测 | 待确认 |
| MTK NPU INT8 | 待测 | 待测 | 待确认 |

普通 COCO 17 点 person keypoints AP 与 133 点 COCO-WholeBody AP 不是同一指标,不得混写.
