# YOLOv5s 性能报告

状态：已完成板端实测（2026-09-07）。

2026-09-07 实跑记录: 旧 MDLA 3.0 DLA 被板端 Runtime 拒绝; MDLA 5.3 直编译因
NCC 将 “MDLA → Output 数据转换桥” 派发到 EDPA_1_2 且 MT8189 无 EDPA 硬件而加载失败。
改用 `--arch=mdla5.3 --suppress-output --disallow-bridge` 后板端推理成功，
输出为 MDLA 原生 NCHW INT8 (行 stride 16 对齐)，由后处理还原布局。
详见模型 README “MT8189 编译约束”。

| 项目 | 数值 |
| --- | --- |
| 设备 | MediaTek Genio 720 EVK |
| 输入 | 1×3×640×640 RGB |
| NeuroPilot SDK | 8.0.11 |
| ncc-tflite | 8.2.31 |
| neuronrt | 8.2.16 |
| 纯 NPU 延迟 | 9.957 ms (100 次连续推理, turbo, 99.2 FPS) |
| 端到端延迟 | 单次进程含 DLA 加载与文件 I/O 97 ms; 常驻会话模式约 10 ms 推理 |
| 峰值内存 | neuronrt 进程 VmHWM 15.5 MB (不含 NPU ION 预留内存) |
| DLA SHA-256 | `cf5b66c3fc1c24c9ef1d5c579d20f8d145cbd15d3a5be874b114f7f270c824c6` |

2026-09-07 实测方式: `deploy/deploy_board.sh` 10 次预热 + 100 次连续推理取 neuronrt
统计; 单次端到端与峰值内存由板端 `/proc/<pid>/status` VmHWM 密集采样获得
(`/usr/bin/time -v` 在该镜像上不可靠, 读数恒为 0)。

## 验收规则

- 先进行预热，再统计固定次数的 P50、P90、P95 和平均延迟。
- 分别记录预处理、纯 NPU、后处理和端到端耗时。
- 同时记录推理进程峰值 RSS；不得把宿主机转换耗时写成板端推理性能。
- 记录板端 NeuroPilot/Neuron Runtime 版本、CPU 调频状态和 DLA SHA-256。
