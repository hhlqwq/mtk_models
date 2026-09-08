# YOLOv5s 精度报告

状态：已完成正式评测（2026-09-08,完整 COCO val2017,5000 张）.

## 评测协议

- 三后端共享同一 letterbox(640×640) 预处理、YOLOv5 解码、逐类 NMS
  (conf 0.001 / IoU 0.6 / max_det 300) 和 pycocotools 评测,清单内全部图片均纳入指标.
- NPU 侧为 MDLA 5.3 `--suppress-output` 原生 NCHW INT8 输出（行 stride 16 对齐）
  加 dequantize；PyTorch/ONNX 为 FP32.
- Ultralytics YOLOv5 v7.0 公布的 YOLOv5s 640 单模型、单尺度 COCO val2017
  mAP@0.5:0.95 为 0.374；0.490 是 YOLOv5l 的指标,不是 YOLOv5s.
- 本评测使用 NMS IoU 0.6,官方复现命令使用 0.65,因此不能作为完全相同协议的官方复现；
  三个后端共享同一协议,后端间精度损失对比仍然有效.

## 结果 (COCO val2017)

| 后端 | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 | AP_small | AP_medium | AP_large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 PyTorch FP32 | 0.3708 | 0.5648 | 0.3970 | 0.2118 | 0.4209 | 0.4820 |
| FP32 ONNX | 0.3709 | 0.5648 | 0.3972 | 0.2119 | 0.4204 | 0.4822 |
| MTK NPU INT8（89 后处理） | 0.3588 | 0.5579 | 0.3861 | 0.2071 | 0.4088 | 0.4627 |
| MTK NPU INT8（板端 C++ 前后处理） | **0.3586** | **0.5579** | **0.3862** | **0.2072** | **0.4087** | **0.4623** |

## 精度损失

| 对比 | mAP@0.5:0.95 | mAP@0.5 |
| --- | ---: | ---: |
| ONNX 相对 PyTorch | +0.0001 | 0.0000 |
| 板端 C++ NPU INT8 相对 ONNX | **-0.0123 (-1.23pt, 相对 -3.3%)** | -0.0069 (-0.69pt) |
| 板端 C++ 相对原 89 后处理路径 | -0.0002 (-0.02pt) | 0.0000 |

结论：在本项目统一评测协议下,板端 C++ 全流程的 INT8 PTQ 损失 1.23pt
mAP@0.5:0.95.板端 C++ 与原 89 后处理路径只相差 0.02pt,可视为 JPEG 缩放和浮点实现
细节造成的微小差异；两条路径的结论一致.
PyTorch 基线 0.3708 与上游公布的 0.374 接近；是否接受 INT8 精度损失应由具体业务阈值决定.

## 证据

- 评测命令：`bash deploy/accuracy_eval.sh all`（89 宿主机）.
- 数据集：`/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017`.
- 本次原始产物：`.eval/yolov5s/{npu,torch,onnx}_results.jsonl`、`*_summary.json`、
  `full_eval` 日志 `/tmp/hailongcodex/2026-09-07/full_eval.log`.
- 后续评测脚本使用 `.eval/yolov5s/runs/<run_id>/` 隔离每次运行,避免旧结果污染.
- 样本数：5000/5000 图片；NPU、PyTorch、ONNX 分别为 718,881、709,477、709,518 条检测结果.
- 最终交付运行 ID：`20260908_cpp_delivery_v3`；5000/5000 图片,718,891 条检测结果.
- 板端原始证据：
  `/root/hailong.he/yolov5s_cpp/runs/20260908_cpp_delivery_v3/`；89 回传的完整证据：
  `examples/output/board_cpp_accuracy/20260908_cpp_delivery_v3/`.逐图 NPU 原始输出未回传.
- 输入清单绑定 Git 提交 `1dcadbe4a5dec82b2e95efd3e0ffea90354a6bfc`、源码、板端二进制、
  DLA、评测器、COCO 标注和 5000 张图片哈希清单；输出清单记录正式结果文件 SHA-256.
- 板端 C++ 精确 AP@0.5:0.95 为 `0.35859860348732847`,评测脚本会先验证
  `processed_ids.txt` 与 COCO 标注中的 5000 个 image_id 完全一致,再运行 COCOeval.
- 五个模型产物的 SHA-256 见 `model_card.md`,板端 DLA 记录见
  `examples/output/SHA256SUMS`.后续转换和编译脚本会重新生成 `models/SHA256SUMS`.
