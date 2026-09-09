# RTMPose Body2d 精度报告

状态：Genio 720 板端 INT8 正式评测已完成.

| 后端 | WholeBody AP | WholeBody AR | 验证配置 |
| --- | ---: | ---: | --- |
| 来源 PyTorch | 待测 | 待测 | 与下述协议相同，尚未执行 |
| FP32 ONNX | 待测 | 待测 | 与下述协议相同，尚未执行 |
| MTK NPU INT8 | 0.4369 | 0.5646 | `20260909_wholebody_int8_v2` |

普通 COCO 17 点 person keypoints AP 与 133 点 COCO-WholeBody AP 不是同一指标,不得混写.

## 正式 COCO-WholeBody 指标

| 部位 | AP | AP50 | AP75 | AR |
| --- | ---: | ---: | ---: | ---: |
| body | 0.5987 | 0.8358 | 0.6653 | 0.6863 |
| foot | 0.5558 | 0.7762 | 0.6036 | 0.7101 |
| face | 0.6534 | 0.9273 | 0.7361 | 0.7692 |
| left hand | 0.2753 | 0.5568 | 0.2501 | 0.4100 |
| right hand | 0.2726 | 0.5639 | 0.2398 | 0.4082 |
| wholebody | 0.4369 | 0.7750 | 0.4362 | 0.5646 |

评测使用 COCO-WholeBody V1.0 验证集 5,000 张图、MMPose 提供的 Faster R-CNN
104,125 个人体检测框、`bbox_keypoint` 重评分、关键点阈值 0.2、WholeBody OKS-NMS
阈值 0.9 和 `xtcocotools` COCOeval.NMS 后保留 89,565 个结果，覆盖 3,893 张
存在检测框的图片.

正式运行的板端常驻 C++ 耗时为：预处理平均 2.7740 ms、NPU 平均 3.8500 ms、后处理
平均 1.0524 ms，峰值 RSS 35,756 KB.这是逐检测框耗时，不包含上游人体检测器.

原始证据保存在 89 服务器：

```text
/data/users/hailong.he/github/mtk_models/models/interaction/pose_detection/rtmpose_body2d/examples/output/board_cpp_accuracy/20260909_wholebody_int8_v2
```

关键证据 SHA-256：`predictions.jsonl` 为
`19e8a8d6003050797a6cb3039cf7e884e6d41bb82fdfa027cadbdf19be2ac0ea`，指标 JSON 为
`a56ba932ca0828be0ebb26ed7646e4c30c82101a8ffbe38d257a2c61c393b1c2`，完整输出哈希
记录在运行目录的 `run_outputs_sha256.txt`.

## 板端双图一致性检查

2026-09-09 使用两个不同 COCO person 框比较同一输入下的 FP32 ONNX 与 MTK NPU 输出:

| 样例 | 全 133 点 argmax 一致率 | ONNX 分数 ≥0.1 | ONNX 分数 ≥0.2 |
| --- | ---: | ---: | ---: |
| sample_1 | 10.53% | 130 点，均值误差 2.463 px | 125 点，均值误差 1.023 px，最大 4.950 px |
| sample_2 | 15.79% | 131 点，均值误差 1.601 px | 123 点，均值误差 0.827 px，最大 3.000 px |

修正输入像素量纲并重新校准后，INT8 SimCC 峰值相对 FP32 ONNX 存在量化位移，但
高置信度关键点的输入空间平均误差约 1 px.双图检查仅用于后端数值诊断；正式结论以上述
完整 COCO-WholeBody AP/AR 为准.
