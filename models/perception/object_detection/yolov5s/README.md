# YOLOv5s · Genio 720

COCO 80 类目标检测。模型来源和许可证见[模型卡](model_card.md)。

## 第一步：编译并上传

在 [run.sh](deploy/run.sh) 顶部填写路径。通常只需确认 `MODEL_WEIGHTS`、填写 `CALIBRATION_DIR` 和 `BOARD_DEPLOY_DIR`，并按需调整 `MODEL_OUTPUT_DIR`、`OUTPUT_DLA`。全量测试再填写 `BOARD_DATASET_DIR`。其余配置仅在 Docker、SDK 或交叉编译环境变化时修改。资源地址见 [source_url.txt](models/source_url.txt)。模型、校准集及产物路径必须同时对编译主机和 Docker 容器可见。

在**编译主机的仓库根目录**运行：

```bash
bash models/perception/object_detection/yolov5s/deploy/run.sh
```

脚本在 Docker 中生成 DLA，在编译主机交叉编译板端 C++ 程序，然后上传 DLA、程序、本脚本及三张示例图。源码、补丁和临时 C++ 程序使用 `/tmp/hailongcodex/当天日期/yolov5s/`，其中 Docker 与编译主机的临时目录各自独立；模型产物写入 `MODEL_OUTPUT_DIR`。临时文件保留供排查，可按需清理。脚本不下载模型或数据。

## 第二步：开发板测试

使用脚本中配置的 `BOARD_HOST` 登录开发板，并执行第一步打印的 `[NEXT]` 命令。

默认测试三张示例图；在板端命令末尾加 `full` 执行 COCO 全量测试。结果写入 `BOARD_RESULTS_DIR`，默认位于 `BOARD_DEPLOY_DIR/results`。全量测试要求 `BOARD_DATASET_DIR/images/` 中有 5000 张 COCO val2017 图片、`BOARD_DATASET_DIR/annotations/instances_val2017.json`，板端还需 `pycocotools`。若 SSH 提示主机密钥变化，先核对板端指纹，再更新编译主机上执行脚本用户的 `known_hosts`。

## 全量测试结果

全量测试成功后,每次运行目录只保留一个 `summary.json`。不再生成 `report/` 副本、系统快照和输入清单; 原始预测、逐图耗时及日志在汇总成功后删除。测试失败时保留中间文件,历史结果不会自动清理。

终端和 `summary.json` 仅汇总板端 NPU 平均推理耗时、FP32 与 INT8 的 mAP@0.5:0.95 和精度下降百分点。NPU 耗时取本次 5000 张图片的 `npu_ms.mean`,不包含预热、图片读取、预处理、后处理和 COCOeval。精度下降按 `(FP32 - INT8) × 100` 计算,负数表示精度提升。

脚本顶部的 `FP32_MAP` 默认使用历史同协议 ONNX FP32 基准 `0.3709`,输出会明确标注历史来源,不是本次 FP32 实测。更换权重或评测协议后,必须替换为匹配的 FP32 基准并更新 `FP32_BASELINE_SOURCE`;没有匹配基准时把 `FP32_MAP` 留空,报告不计算损失。比较包含转换、量化与板端执行的整体精度变化。

## 历史精度

2026-09-08 在 COCO val2017 全量 5000 张图片上完成评测。PyTorch、ONNX 和 NPU 使用同一 letterbox 640×640、YOLOv5 解码、逐类 NMS 和 COCOeval 协议，阈值为 conf 0.001、IoU 0.6、max_det 300。NPU 使用 MDLA 5.3 的原生 NCHW INT8 输出，行 stride 为 16，后处理负责反量化。该协议的 NMS IoU 与上游公布结果的 0.65 不同，因此下表适用于后端间对照。

| 后端 | mAP@0.5:0.95 |
| --- | ---: |
| PyTorch FP32 | 0.3708 |
| ONNX FP32 | 0.3709 |
| MTK NPU INT8，编译主机后处理 | 0.3588 |
| MTK NPU INT8，板端 C++ 前后处理 | **0.3586** |

板端 C++ INT8 相对 ONNX 的 mAP@0.5:0.95 下降 0.0123，即 1.23 个百分点；相对编译主机后处理路径下降 0.0002。历史交付运行 ID 为 `20260908_cpp_delivery_v3`，板端处理 5000/5000 张图片，得到 718891 条检测结果，精确 AP@0.5:0.95 为 `0.35859860348732847`。当时的原始运行目录已不在当前工作树中；以上数值是历史记录，不代表本次改脚本后已经重测。

## 历史性能

2026-09-08 在 Genio 720 EVK、NeuroPilot SDK 8.0.11、ncc-tflite 8.2.31、neuronrt 8.2.16 上完成板端 C++ 测量。编译使用 `--arch=mdla5.3 --suppress-output --disallow-bridge`；板端预热 20 次，再统计 5000 张图片。稳态端到端包含 JPEG 读取、letterbox、RGB 转换、INT8 量化、NPU 推理、解码和 NMS，不含模型加载、最终 JSON 写入和 COCOeval。

| 项目 | 历史结果 |
| --- | ---: |
| 纯 NPU，100 次连续推理 | 平均 9.957 ms，99.2 FPS |
| C++ 预处理 | 平均 7.716 ms；P50/P90/P95 为 7.196/10.189/11.039 ms |
| C++ NPU | 平均 9.629 ms；P50/P90/P95 为 9.629/9.687/9.696 ms |
| C++ 后处理 | 平均 16.173 ms；P50/P90/P95 为 16.058/17.208/17.662 ms |
| C++ 稳态端到端 | 平均 33.662 ms，约 29.71 FPS；P50/P90/P95 为 33.116/36.628/38.009 ms |
| 进程峰值内存 | 33224 KiB，约 32.45 MiB |

CPU 调频策略为 `schedutil`，`policy0` 范围 500 MHz–2.0 GHz，`policy6` 范围 550 MHz–2.6 GHz。历史板端 RTC 未同步，原系统快照显示 2025-08-01；评测日期以运行 ID 和编译主机发起日期为准。历史汇总保留在 `examples/output/timing_summary_current_run.json`。

## 示例图片

`examples/input/` 中的三张图片由项目维护者于 2026-09-11 使用 OpenAI 图像生成工具生成，不取自 COCO 或 ImageNet。项目维护者按 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) 发布，仅用于板端展示，不参与正式精度评测。
