# YOLOv5s 精度报告

状态：已完成正式评测（2026-09-07，完整 COCO val2017，5000 张）。

## 评测协议

- 三后端共享同一 letterbox(640×640) 预处理、YOLOv5 解码、逐类 NMS
  (conf 0.001 / IoU 0.6 / max_det 300) 和 pycocotools 评测（imgIds 限定为已推理图片）。
- NPU 侧为 MDLA 5.3 `--suppress-output` 原生 NCHW INT8 输出（行 stride 16 对齐）
  加 dequantize；PyTorch/ONNX 为 FP32。
- 注意：固定 640×640 方形输入的 mAP 低于 yolov5 官方 rect 多尺度协议 (0.490)，
  但三后端协议完全一致，精度损失对比有效。

## 结果 (COCO val2017)

| 后端 | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 | AP_small | AP_medium | AP_large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 PyTorch FP32 | 0.3708 | 0.5648 | 0.3970 | 0.2118 | 0.4209 | 0.4820 |
| FP32 ONNX | 0.3709 | 0.5648 | 0.3972 | 0.2119 | 0.4204 | 0.4822 |
| MTK NPU INT8 | 0.3588 | 0.5579 | 0.3861 | 0.2071 | 0.4088 | 0.4627 |

## 精度损失

| 对比 | mAP@0.5:0.95 | mAP@0.5 |
| --- | ---: | ---: |
| ONNX 相对 PyTorch | +0.0001 | 0.0000 |
| NPU INT8 相对 ONNX | **-0.0121 (-1.20pt, 相对 -3.3%)** | -0.0069 (-0.69pt) |

结论：INT8 PTQ 损失 1.2pt mAP@0.5:0.95，处于 YOLOv5s INT8 量化正常区间
（业界同类交付通常 <2pt），**精度损失不大，无需修复**。

## 证据

- 评测命令：`bash deploy/accuracy_eval.sh all`（89 宿主机）。
- 数据集：`/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017`。
- 原始产物：`.eval/yolov5s/{npu,torch,onnx}_results.jsonl`、`*_summary.json`、
  `full_eval` 日志 `/tmp/hailongcodex/2026-09-07/full_eval.log`。
- 样本数：5000/5000 图片，三后端各自约 30 万级检测框。
- DLA SHA-256 见 `models/SHA256SUMS`（板端记录见 `examples/output/SHA256SUMS`）。
