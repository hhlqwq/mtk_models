# RTMPose Body2d 精度报告

## 正式开源模型三后端结果

2026-09-14 完成 OpenMMLab MMPose v1.3.2 官方 RTMPose-M 的 PyTorch FP32、
自行导出 ONNX FP32 和 Genio 720 MTK NPU INT8 三后端同协议对比.PyTorch/ONNX
运行 ID 为 `20260914_rtmpose_three_backend_v1`,板端运行 ID 为
`20260910_mmpose_official_v1`.三个后端均使用同一份 MMPose Faster R-CNN 的
104,125 个人体检测框、同一套 192×256 RGB 仿射预处理、SimCC 解码、
`bbox_keypoint` 重评分、0.2 关键点阈值和 0.9 WholeBody OKS-NMS.

| 后端 | WholeBody AP | AP50 | AP75 | AR | AP 相对 PyTorch |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP32 | 0.5702 | 0.8422 | 0.6382 | 0.6654 | +0.0000 |
| ONNX FP32 | 0.5703 | 0.8422 | 0.6381 | 0.6656 | +0.0002 |
| MTK NPU INT8 | 0.5324 | 0.8366 | 0.5903 | 0.6413 | -0.0378 |

PyTorch 与 ONNX 的 WholeBody AP 仅相差 0.0002,说明自行导出 ONNX 未产生可见的
任务指标回退.NPU INT8 相对 ONNX FP32 的 WholeBody AP 下降 0.0380,即 3.80 个
百分点；AP50 下降 0.0056,AP75 下降 0.0478.因此本次量化损失主要体现在更严格的
关键点定位精度,而不是宽松阈值下是否检测到关键点.

| 部位 | PyTorch AP | ONNX AP | NPU INT8 AP | NPU 相对 ONNX |
| --- | ---: | ---: | ---: | ---: |
| body | 0.6659 | 0.6661 | 0.6569 | -0.0092 |
| foot | 0.6020 | 0.6018 | 0.5845 | -0.0174 |
| face | 0.8127 | 0.8129 | 0.7271 | -0.0858 |
| left hand | 0.4629 | 0.4627 | 0.4139 | -0.0487 |
| right hand | 0.4452 | 0.4447 | 0.3966 | -0.0481 |
| wholebody | 0.5702 | 0.5703 | 0.5324 | -0.0380 |

分部结果显示 face 和双手对 INT8 量化最敏感.这是基于本次指标差异得到的结论,
后续若优化量化策略,应优先检查小尺度脸部和手部 SimCC 峰值保持情况.

官方模型页公布的 PyTorch WholeBody AP/AR 为 0.582/0.674.本项目同协议实测为
0.5702/0.6654,分别低 0.0118 和 0.0086.由于正式转换损失必须排除评测实现与运行环境
差异,本项目以同一评测程序得到的 PyTorch 0.5702 作为 ONNX 和 INT8 的直接基线,
官方公布值只作为外部参考.

完整运行共覆盖 3,893 张存在检测框的图片.PyTorch、ONNX 和 NPU 在 OKS-NMS 后分别
保留 84,709、84,704 和 87,016 个结果.输入、预测、指标和汇总文件 SHA-256 全部
校验通过.紧凑证据见 `accuracy_comparison_20260914.json`.

PyTorch/ONNX 耗时来自宿主机 RTX 4090 D,NPU 耗时来自 Genio 720,只能分别说明各自
运行环境中的执行情况,不能作为同设备性能对比.

89 原始证据位于
`models/interaction/pose_detection/rtmpose_body2d/examples/output/board_cpp_accuracy/20260910_mmpose_official_v1/`.
指标 JSON SHA-256 为
`706b1f3119c14958fc969d2df0f0d5d943f3d22cf81afa3ec78867eb8a6c3b67`.
当前 DLA SHA-256 为
`a5851c2e9602a17f703dbcaa2a80fd01e13bd99c11d7b2437a1e286374ed1461`.

PyTorch/ONNX 全量原始证据保存在 89 服务器：

```text
/data/users/hailong.he/github/mtk_models/.eval/rtmpose_body2d/runs/20260914_rtmpose_three_backend_v1
```

> 历史结果说明：本页结果来自 Qualcomm v0.61.0 预导出 FP32 ONNX 及其 MTK INT8
> 衍生模型.它们保留用于工程对照,不属于当前已锁定的 OpenMMLab MMPose 上游正式交付链路.

历史状态：Genio 720 板端 INT8 正式评测已完成.

| 后端 | WholeBody AP | WholeBody AR | 验证配置 |
| --- | ---: | ---: | --- |
| 历史来源 PyTorch | 未执行 | 未执行 | Qualcomm 预导出模型未提供此基线 |
| 历史 FP32 ONNX | 未执行 | 未执行 | 历史流程未完成同协议全量评测 |
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
阈值 0.9 和 `xtcocotools` COCOeval.NMS 后保留 89,565 个结果,覆盖 3,893 张
存在检测框的图片.

正式运行的板端常驻 C++ 耗时为：预处理平均 2.7740 ms、NPU 平均 3.8500 ms、后处理
平均 1.0524 ms,峰值 RSS 35,756 KB.这是逐检测框耗时,不包含上游人体检测器.

原始证据保存在 89 服务器：

```text
/data/users/hailong.he/github/mtk_models/models/interaction/pose_detection/rtmpose_body2d/examples/output/board_cpp_accuracy/20260909_wholebody_int8_v2
```

关键证据 SHA-256：`predictions.jsonl` 为
`19e8a8d6003050797a6cb3039cf7e884e6d41bb82fdfa027cadbdf19be2ac0ea`,指标 JSON 为
`a56ba932ca0828be0ebb26ed7646e4c30c82101a8ffbe38d257a2c61c393b1c2`,完整输出哈希
记录在运行目录的 `run_outputs_sha256.txt`.

## 板端双图一致性检查

2026-09-09 使用两个不同 COCO person 框比较同一输入下的 FP32 ONNX 与 MTK NPU 输出:

| 样例 | 全 133 点 argmax 一致率 | ONNX 分数 ≥0.1 | ONNX 分数 ≥0.2 |
| --- | ---: | ---: | ---: |
| sample_1 | 10.53% | 130 点,均值误差 2.463 px | 125 点,均值误差 1.023 px,最大 4.950 px |
| sample_2 | 15.79% | 131 点,均值误差 1.601 px | 123 点,均值误差 0.827 px,最大 3.000 px |

修正输入像素量纲并重新校准后,INT8 SimCC 峰值相对 FP32 ONNX 存在量化位移,但
高置信度关键点的输入空间平均误差约 1 px.双图检查仅用于后端数值诊断；正式结论以上述
完整 COCO-WholeBody AP/AR 为准.
