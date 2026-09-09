# RTMPose Body2d

## 模型信息

```text
模型: RTMPose-Body2d
任务: WholeBody 人体姿态估计
输入: 1×3×256×192 RGB
输出: 133 个关节点的位置和置信度
设备: MediaTek Genio 720 EVK
当前状态: 环境建设中（等待本轮真实转换与板端验证）
```

RTMPose 是 top-down 姿态模型,只处理人体检测框.本项目使用 COCO person 标注框模拟上游
检测器输出,执行仿射裁剪、INT8 量化、板端推理和 SimCC 解码.完整图片应用仍需额外的人体
检测器；模型自身延迟和“检测器 + RTMPose"端到端延迟必须分别报告.

## 执行流程

```bash
cd /workspace/models/interaction/pose_detection/rtmpose_body2d
./deploy/download_original.sh
./deploy/convert.sh
./deploy/build.sh
./deploy/deploy_board.sh
```

`download_original.sh` 支持复用 `models/` 中已有的官方 ZIP,保留 FP32 ONNX 外部权重,并
生成 MTK Converter 可读取的单文件兼容副本.`convert.sh` 默认使用 100 个 COCO person 框
校准；`build.sh` 固定使用
Genio 720 所需的 `mdla5.3 + --suppress-output + --disallow-bridge`；部署脚本使用两张不同
图片冒烟并采集 20 次预热、100 次连续推理、峰值内存及单次进程耗时.

## 交付状态

| 环节 | 状态 | 证据 |
| --- | --- | --- |
| Hugging Face 来源 | 已锁定 | `original/source_url.txt` |
| Qualcomm FP32 ONNX | 待下载 | `models/model_fp32.onnx` |
| MTK INT8 TFLite | 待执行 | `models/model_int8.tflite` |
| DLA | 待执行 | `models/model_int8.dla` |
| 板端 Demo | 待执行 | `examples/output/` |
| WholeBody AP | 等待确认数据集 | `docs/accuracy.md` |
| 板端性能 | 待执行 | `docs/benchmark.md` |

## 验证边界

- 双图板端冒烟用于确认 DLA 可运行、输出完整且不同输入不会得到完全相同的旧缓冲结果.
- 当前 `instances_val2017.json` 只提供 person 框,不能产生正式 133 点 WholeBody AP.
- 完整精度评测必须另行提供 COCO-WholeBody 标注并锁定人体框来源和评测协议.
