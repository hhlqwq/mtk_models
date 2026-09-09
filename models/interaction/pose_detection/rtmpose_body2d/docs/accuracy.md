# RTMPose Body2d 精度报告

状态：正式评测数据和人体框来源尚未确认.

| 后端 | WholeBody AP | WholeBody AR | 验证配置 |
| --- | ---: | ---: | --- |
| 来源 PyTorch | 待测 | 待测 | 待确认 |
| FP32 ONNX | 待测 | 待测 | 待确认 |
| MTK NPU INT8 | 待测 | 待测 | 待确认 |

普通 COCO 17 点 person keypoints AP 与 133 点 COCO-WholeBody AP 不是同一指标,不得混写.

## 板端双图一致性检查

2026-09-09 使用两个不同 COCO person 框比较同一输入下的 FP32 ONNX 与 MTK NPU 输出:

| 样例 | 全 133 点 argmax 一致率 | ONNX 分数 ≥0.1 | ONNX 分数 ≥0.2 |
| --- | ---: | ---: | ---: |
| sample_1 | 92.48% | 116 点，95.69% | 49 点，100%，均值误差 0 px |
| sample_2 | 75.94% | 88 点，94.32% | 41 点，100%，均值误差 0 px |

样例 2 的 ONNX 平均分仅 0.1514；低置信度 SimCC 峰值存在明显量化漂移.这项检查证明
板端输出可用且高置信度峰值与 ONNX 一致,但不能替代 COCO-WholeBody AP.正式 AP/AR
继续标记为未验证.
