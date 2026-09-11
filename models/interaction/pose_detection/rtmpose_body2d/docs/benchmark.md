# RTMPose Body2d 性能报告

## 正式开源模型结果

运行 ID `20260910_mmpose_official_v1` 在 Genio 720 上完成 104,125 个检测框的常驻
C++ 全量运行.

| 项目 | 数值 |
| --- | ---: |
| 平均预处理耗时 | 2.7658 ms/框 |
| 平均 NPU 耗时 | 3.8527 ms/框 |
| 平均后处理耗时 | 1.0473 ms/框 |
| 三阶段合计 | 7.6657 ms/框 |
| 峰值 RSS | 36,732 KB |

该数据是单个人体检测框的 RTMPose 耗时,不包含上游人体检测器.正式 DLA SHA-256 为
`a5851c2e9602a17f703dbcaa2a80fd01e13bd99c11d7b2437a1e286374ed1461`,
时间摘要 SHA-256 为
`75e987711576deb7b9fa3e657348d6ec1eae625cc35ff3d2a964528373f53817`.

> 历史结果说明：本页数据来自 Qualcomm v0.61.0 预导出 FP32 ONNX 的 MTK 衍生模型,
> 不属于当前已锁定的 OpenMMLab MMPose 上游正式交付链路.

| 项目 | 数值 |
| --- | --- |
| 设备 | MediaTek Genio 720 EVK |
| 输入 | 1×3×256×192 BGR INT8 |
| 输出 | 133 个关节点 |
| NeuroPilot SDK | 8.0.11 |
| ncc-tflite | 8.2.31 |
| neuronrt | 8.2.16 |
| RTMPose 纯 NPU 延迟 | 3.76894 ms/inf |
| RTMPose 纯 NPU 吞吐 | 260 FPS |
| 单次进程端到端耗时 | 91 ms |
| 检测器 + RTMPose 延迟 | 待测 |
| neuronrt 峰值 RSS | 27,392 KB |
| 正式 C++ 流程 NPU 平均耗时 | 3.84997 ms/框 |
| 正式 C++ 流程总平均耗时 | 7.67634 ms/框 |
| 正式 C++ 流程峰值 RSS | 35,756 KB |

## 测试配置

- 日期：2026-09-09.
- 设备：Genio 720 EVK，Linux 6.6.117-mtk，aarch64.
- DLA SHA-256：`0b440fb98f7a39ee8c25b7251beb96651c2f54c177bf3d6ef80e81525fcb99c1`.
- 编译参数：`--arch=mdla5.3 --suppress-output --disallow-bridge`.
- 性能方法：先独立预热 20 次，再由 `neuronrt -c 100 -b 100 -r turbo` 连续推理
  100 次；总推理时间 376.894 ms.
- 单次进程耗时包含 neuronrt 进程启动、模型加载、一次推理和退出，不含上游人体检测器.
- 峰值 RSS 来自 100 次连续推理期间采样 `/proc/<pid>/status` 的 `VmHWM`.
- 正式 C++ 流程来自 104,125 个检测框的完整运行，平均预处理 2.77401 ms、NPU
  3.84997 ms、后处理 1.05235 ms；不包含上游人体检测器和图片检测耗时.

完整机器可读证据见 `board_validation_20260909.json`.
