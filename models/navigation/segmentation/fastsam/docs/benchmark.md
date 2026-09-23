# Genio 720 性能

当前没有 DLA 产物或板端性能结果,不填写估算 FPS.

C++ Demo 单独记录:

- `preprocess_ms`: C++ JPEG 预处理与量化耗时.
- `model_load_ms`: 模型加载和 IO 契约校验耗时.
- `npu_ms`: 已加载 DLA 后单次 `NeuronRuntime_inference` 调用墙钟耗时.
- `postprocess_ms`: CPU DFL、NMS、掩码还原与提示选择耗时.
- `end_to_end_ms`: 从图片读取至提示选择完成,包含模型加载,不含输出文件写入.
- `peak_rss_kib`: C++ 进程峰值 RSS.
- `smoke.log`: 板端 C++ 单次运行日志.

本次为单次冒烟,没有预热后重复推理统计.`npu_ms` 包含 API 调用边界,
不能直接宣称是纯 MDLA 核内延迟.正式基准需在同一 Runtime 实例中预热后重复测量.
正式发布需同时记录板端系统、Runtime 版本、模型哈希、电源/性能档位与统计方法.
