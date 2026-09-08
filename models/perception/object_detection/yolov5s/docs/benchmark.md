# YOLOv5s 性能报告

状态：已完成板端 C++ 稳态端到端、峰值内存和 CPU 调频状态实测（2026-09-08）.

2026-09-07 实跑记录: 旧 MDLA 3.0 DLA 被板端 Runtime 拒绝; MDLA 5.3 直编译因
NCC 将 “MDLA → Output 数据转换桥" 派发到 EDPA_1_2 且 MT8189 无 EDPA 硬件而加载失败.
改用 `--arch=mdla5.3 --suppress-output --disallow-bridge` 后板端推理成功,
输出为 MDLA 原生 NCHW INT8 (行 stride 16 对齐),由后处理还原布局.
详见模型 README “MT8189 编译约束".

| 项目 | 数值 |
| --- | --- |
| 设备 | MediaTek Genio 720 EVK |
| 输入 | 1×3×640×640 RGB |
| NeuroPilot SDK | 8.0.11 |
| ncc-tflite | 8.2.31 |
| neuronrt | 8.2.16 |
| 纯 NPU 平均延迟 | 9.957 ms（100 次连续推理,turbo,99.2 FPS） |
| 10 次预热推理 | 总计 97.311 ms,平均 9.731 ms/次；不是单次端到端延迟 |
| 板端 C++ 预处理 | 平均 7.716 ms；P50/P90/P95 = 7.196/10.189/11.039 ms |
| 板端 C++ NPU | 平均 9.629 ms；P50/P90/P95 = 9.629/9.687/9.696 ms |
| 板端 C++ 后处理 | 平均 16.173 ms；P50/P90/P95 = 16.058/17.208/17.662 ms |
| 板端 C++ 稳态端到端 | 平均 33.662 ms（约 29.71 FPS）；P50/P90/P95 = 33.116/36.628/38.009 ms |
| 峰值内存 | 33,224 KiB（34,021,376 bytes,约 32.45 MiB）；由进程 `getrusage(RUSAGE_SELF).ru_maxrss` 持久化记录 |
| CPU 调频 | `policy0`、`policy6` 运行前后均为 `schedutil`；频率范围分别为 500 MHz–2.0 GHz、550 MHz–2.6 GHz |
| DLA SHA-256 | `cf5b66c3fc1c24c9ef1d5c579d20f8d145cbd15d3a5be874b114f7f270c824c6` |

2026-09-08 使用 `deploy/accuracy_board_cpp.sh` 在 92 上完整处理 COCO val2017 5000 张图片.
C++ 程序常驻加载 DLA,20 次预热后逐图记录耗时；预处理包含板端 JPEG 文件读取、letterbox、
RGB 转换和 INT8 量化,后处理包含输出还原、解码、NMS 和坐标回映.稳态端到端统计不包含
一次性模型加载、最终 JSON 序列化及 COCOeval.最终交付运行 ID 为
`20260908_cpp_delivery_v3`,逐图 CSV 和汇总 JSON 位于 89 的
`examples/output/board_cpp_accuracy/20260908_cpp_delivery_v3/`.输入清单绑定运行提交、源码、
板端二进制、DLA、评测器、标注和图片清单哈希；输出清单绑定全部正式结果.

板端 RTC 未同步,`system_before.txt` 和 `system_after.txt` 内的采集时间错误显示为
2025-08-01；运行日期以 89 发起日期和运行 ID 为准.CPU governor、频率范围、内核和系统版本
均直接来自这两个原始快照,不依赖板端墙钟.当前交付状态为“完整交付".

## 验收规则

- 先进行预热,再统计固定次数的 P50、P90、P95 和平均延迟.
- 分别记录预处理、纯 NPU、后处理和端到端耗时.
- 同时记录推理进程峰值 RSS；不得把宿主机转换耗时写成板端推理性能.
- 记录板端 NeuroPilot/Neuron Runtime 版本、CPU 调频状态和 DLA SHA-256.
