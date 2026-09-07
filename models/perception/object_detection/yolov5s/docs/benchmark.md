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
| 纯 NPU 平均延迟 | 9.957 ms（100 次连续推理,turbo,99.2 FPS） |
| 10 次预热推理 | 总计 97.311 ms，平均 9.731 ms/次；不是单次端到端延迟 |
| 端到端延迟 | 待补测：需覆盖预处理、模型加载、文件 I/O、NPU 推理和后处理 |
| P50/P90/P95 | 待补测：当前 Runtime 日志仅提供总时间和平均值 |
| 峰值内存 | 待补测：`/usr/bin/time -v` 在当前板端镜像中报告 0，已有 15.5 MB 口头记录缺少持久化采样日志 |
| DLA SHA-256 | `cf5b66c3fc1c24c9ef1d5c579d20f8d145cbd15d3a5be874b114f7f270c824c6` |

2026-09-07 实测方式：`deploy/deploy_board.sh` 执行 10 次预热和 100 次连续推理，以上纯 NPU
平均延迟来自 neuronrt 汇总日志。当前保存的产物不能证明单次端到端延迟、延迟分位数或
15.5 MB 峰值内存，因此这些项目保留为待补测，不纳入已完成结论。

当前交付状态为“板端已验证”。完成下列验收规则并保存原始日志后，才能升级为“完整交付”。

## 验收规则

- 先进行预热，再统计固定次数的 P50、P90、P95 和平均延迟。
- 分别记录预处理、纯 NPU、后处理和端到端耗时。
- 同时记录推理进程峰值 RSS；不得把宿主机转换耗时写成板端推理性能。
- 记录板端 NeuroPilot/Neuron Runtime 版本、CPU 调频状态和 DLA SHA-256。
