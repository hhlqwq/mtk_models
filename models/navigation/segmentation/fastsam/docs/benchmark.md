# Genio 720 性能

COCO val2017 全量评测已记录逐图耗时；预热后常驻 Runtime 的稳定性能仍待测。

运行编号 `20260928_fastsam_full_v1` 处理 5,000 张图片，`NeuronRuntime_inference` 平均 `14.8848309588 ms/张`，逐图端到端平均 `142.9671195278 ms/张`。[完整报告](../results/full_accuracy/20260928_fastsam_full_v1/summary.json)。shell 对每张图片重新启动 C++ 程序并加载模型，端到端数值包含该开销，不能换算成常驻实例吞吐量。

C++ Demo 单独记录:

- `preprocess_ms`: C++ JPEG 预处理与量化耗时.
- `model_load_ms`: 模型加载和 IO 契约校验耗时.
- `npu_ms`: 已加载 DLA 后单次 `NeuronRuntime_inference` 调用墙钟耗时.
- `postprocess_ms`: CPU DFL、NMS、掩码还原与提示选择耗时.
- `end_to_end_ms`: 从图片读取至提示选择完成,包含模型加载,不含输出文件写入.
- `peak_rss_kib`: C++ 进程峰值 RSS.
- `smoke.log`: 板端 C++ 单次运行日志.

先前单次冒烟没有预热后重复推理统计.`npu_ms` 包含 API 调用边界,
不能直接宣称是纯 MDLA 核内延迟.正式基准需在同一 Runtime 实例中预热后重复测量.
正式发布需同时记录板端系统、Runtime 版本、模型哈希、电源/性能档位与统计方法.

本次单次结果见 [板端冒烟报告](board_smoke_20260923.md):
`NeuronRuntime_inference` 调用 15.909384 ms,端到端 243.595616 ms,
峰值 RSS 135252 KiB.这三个值属于同一张图片、同一次运行,
没有置信区间,不能解释为稳定吞吐量.
