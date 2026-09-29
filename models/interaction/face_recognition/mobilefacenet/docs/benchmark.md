# 性能报告

状态：全量逐图 CLI 耗时与常驻 C++ Neuron Runtime 调用延迟均已记录。

运行编号 `20260928_mobilefacenet_lfw_full_v1`，7,701 张不同图片逐图调用 `neuronrt`，CLI 墙钟平均 `36.2141417901 ms/图`。[完整报告](../results/full_accuracy/20260928_mobilefacenet_lfw_full_v1/summary.json)。该口径包含每图进程启动和模型加载，不是常驻实例的单次 NPU 推理耗时。

运行编号 `20260929_board_persistent_v1` 在 92 使用与 LFW 全量精度相同的 DLA 和真实量化样例 `face_1.bin`，常驻 C++ Neuron Runtime 预热 10 次、正式连续推理 100 次。[机器可读报告](../results/benchmark/20260929_board_persistent_v1/summary.json)。

| 计时范围 | Mean | P50 | P95 | 峰值 RSS |
| --- | ---: | ---: | ---: | ---: |
| `NeuronRuntime_inference` API 调用 | 0.476406 ms | 0.472807 ms | 0.497589 ms | 7,936 KiB |

DLA SHA-256 为 `ff0735dc157f478c5ceed564352c031749bd5d74fa97fb8afca22b3233af3e89`，输入 SHA-256 为 `0e351a24950a02debafde109704726783547b662a22504727d0ac8f46328afc7`。程序在 89 使用 `bash deploy/build_board_benchmark.sh` 交叉编译，在 92 使用 `deploy/benchmark_board.cpp` 的参数 `--model`、`--input`、`--report`、`--warmup 10 --repeats 100` 运行。上述数值只覆盖 Runtime API 调用，不含检测、对齐、特征匹配和多帧决策，也不声称是芯片核内纯计算时间。

2026-09-24 的[板端冒烟](smoke.md)仅执行三次 `neuronrt -m hw` 推理，不作为正式性能或 FPS。应用流水线还需分别统计检测、对齐、特征提取、匹配与多帧决策耗时。
