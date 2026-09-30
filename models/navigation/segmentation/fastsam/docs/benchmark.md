# Genio 720 性能

独立校准 DLA 的 COCO val2017 全量评测已记录逐图耗时；预热后常驻 Runtime 性能也已测得。

运行编号 `20260929_fastsam_imagenet100_full_v2` 对 5,000 张图执行真实 C++ 板端推理，
单次 `NeuronRuntime_inference` 平均 `14.2883 ms/张`，逐图端到端平均 `143.0013 ms/张`。
每张图重新启动程序并加载模型，端到端数值包含该开销。[全量报告](../results/full_accuracy/20260929_fastsam_imagenet100_full_v2/summary.json)。

运行编号 `20260930_board_persistent_imagenet100_v2` 使用同一 DLA 和真实量化输入，
板端常驻 C++ Neuron Runtime 预热 10 次、正式推理 100 次。
[机器可读报告](../results/benchmark/20260930_board_persistent_imagenet100_v2/summary.json)。

| 计时范围 | Mean | P50 | P95 | 峰值 RSS |
| --- | ---: | ---: | ---: | ---: |
| 新 DLA `NeuronRuntime_inference` API 调用 | 14.0536 ms | 14.0501 ms | 14.0874 ms | 21,540 KiB |

新 DLA SHA-256 为 `6aa12c1c68d0eb13ee7669c83c36e838581a1a4ab917e2f73f37378edc13b7e5`，
本次输入 SHA-256 为 `8310f7e5117547b4211c8aa792d41ec006f22ae9485925ec80c9938357030b5f`。
常驻计时只覆盖 Runtime API 调用，不含预处理、模型加载、CPU 掩码后处理及写文件。

## 旧 DLA 历史性能

运行编号 `20260928_fastsam_full_v1` 处理 5,000 张图片，`NeuronRuntime_inference` 平均 `14.8848309588 ms/张`，逐图端到端平均 `142.9671195278 ms/张`。[完整报告](../results/full_accuracy/20260928_fastsam_full_v1/summary.json)。shell 对每张图片重新启动 C++ 程序并加载模型，端到端数值包含该开销，不能换算成常驻实例吞吐量。

C++ Demo 单独记录:

- `preprocess_ms`: C++ JPEG 预处理与量化耗时.
- `model_load_ms`: 模型加载和 IO 契约校验耗时.
- `npu_ms`: 已加载 DLA 后单次 `NeuronRuntime_inference` 调用墙钟耗时.
- `postprocess_ms`: CPU DFL、NMS、掩码还原与提示选择耗时.
- `end_to_end_ms`: 从图片读取至提示选择完成,包含模型加载,不含输出文件写入.
- `peak_rss_kib`: C++ 进程峰值 RSS.
- `smoke.log`: 板端 C++ 单次运行日志.

运行编号 `20260929_board_persistent_v1` 在 92 使用与全量精度相同的 DLA 和真实量化输入 `input_int8.bin`，常驻 C++ Neuron Runtime 预热 10 次、正式连续推理 100 次。[机器可读报告](../results/benchmark/20260929_board_persistent_v1/summary.json)。

| 计时范围 | Mean | P50 | P95 | 峰值 RSS |
| --- | ---: | ---: | ---: | ---: |
| `NeuronRuntime_inference` API 调用 | 14.2735 ms | 14.2641 ms | 14.3713 ms | 20,260 KiB |

DLA SHA-256 为 `d2dc50f4ed65bc4faf2cd929ef262ed102ed610036fee0202ba2b675ecfb98e7`，输入 SHA-256 为 `e48b2c311f81639a4d0ae5c4a10e8544aeacf585286edc1a834b7265c3e157db`。程序在 89 使用 `bash deploy/build_board_benchmark.sh` 交叉编译，在 92 使用 `deploy/benchmark_board.cpp` 的参数 `--model`、`--input`、`--report`、`--warmup 10 --repeats 100` 运行。计时不含预处理、模型加载、CPU 掩码后处理和结果写入；`npu_ms` 包含 Runtime API 边界，不能直接称作 MDLA 核内纯计算时间。电源与性能档位未单独锁定，跨设备比较仍需控制这些条件。

本次单次结果见 [板端冒烟报告](board_smoke_20260923.md):
`NeuronRuntime_inference` 调用 15.909384 ms,端到端 243.595616 ms,
峰值 RSS 135252 KiB.这三个值属于同一张图片、同一次运行,
没有置信区间,不能解释为稳定吞吐量.
