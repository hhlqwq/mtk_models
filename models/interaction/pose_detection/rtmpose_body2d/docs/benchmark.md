# RTMPose Body2d 性能报告

| 项目 | 数值 |
| --- | --- |
| 设备 | MediaTek Genio 720 EVK |
| 输入 | 1×3×256×192 RGB |
| 输出 | 133 个关节点 |
| NeuroPilot SDK | 8.0.11 |
| ncc-tflite | 8.2.31 |
| neuronrt | 8.2.16 |
| RTMPose 纯 NPU 延迟 | 3.73232 ms/inf |
| RTMPose 纯 NPU 吞吐 | 262.3 FPS |
| 单次进程端到端耗时 | 91 ms |
| 检测器 + RTMPose 延迟 | 待测 |
| neuronrt 峰值 RSS | 27,136 KB |

## 测试配置

- 日期：2026-09-09.
- 设备：Genio 720 EVK，Linux 6.6.117-mtk，aarch64.
- DLA SHA-256：`3d65b14be989fb933a9e42db9ba7481b82613be29b6d08cb1ee067e11d3be5dc`.
- 编译参数：`--arch=mdla5.3 --suppress-output --disallow-bridge`.
- 性能方法：先独立预热 20 次，再由 `neuronrt -c 100 -b 100 -r turbo` 连续推理
  100 次；总推理时间 373.232 ms.
- 单次进程耗时包含 neuronrt 进程启动、模型加载、一次推理和退出，不含上游人体检测器.
- 峰值 RSS 来自 100 次连续推理期间采样 `/proc/<pid>/status` 的 `VmHWM`.

完整机器可读证据见 `board_validation_20260909.json`.
