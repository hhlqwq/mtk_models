# 性能评估

运行编号 `20260928_depth_da2k_full_v1` 对 1,033 张 DA-2K 图片逐图调用 `neuronrt -m hw`，CLI 墙钟平均 `194.7004188777 ms/图`。[完整报告](../results/full_accuracy/20260928_depth_da2k_full_v1/summary.json)。该数值包含每图进程启动及模型加载，不是纯 NPU 延迟或常驻模型吞吐量。

运行编号 `20260929_board_persistent_v1` 使用与全量精度相同的 DLA，在 92 板端常驻 C++ Neuron Runtime，使用已有真实量化输入 `image_1.bin`，预热 10 次后连续推理 100 次。[机器可读报告](../results/benchmark/20260929_board_persistent_v1/summary.json)。

| 计时范围 | Mean | P50 | P95 | 峰值 RSS |
| --- | ---: | ---: | ---: | ---: |
| `NeuronRuntime_inference` API 调用 | 133.902 ms | 133.843 ms | 134.417 ms | 33,444 KiB |

DLA SHA-256 为 `8319d41c803e37e50d249fc3a729b007ffe27e214e8af1d4997e4849692e3dce`，输入 SHA-256 为 `4a298dcc12fdbc5c43ef794c49ef2f1efdc1fc232033b199b57f0862200dbff3`。程序在 89 通过 `bash deploy/build_board_benchmark.sh` 交叉编译，在 92 使用 `deploy/benchmark_board.cpp` 的命令参数 `--model`、`--input`、`--report`、`--warmup 10 --repeats 100` 运行。报告不含图片解码、预处理、模型加载、输出还原和文件写入，也不声称是芯片核内纯计算时间。三次早期 `neuronrt -m hw` 冒烟仅用于硬件验证与重复性，见 [smoke.md](smoke.md)。
