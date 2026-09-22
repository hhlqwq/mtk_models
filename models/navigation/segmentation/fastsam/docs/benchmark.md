# Genio 720 性能

当前没有 DLA 产物或板端性能结果,不填写估算 FPS.

Demo 单独记录:

- `inference_wall_ms_including_load`: 模型加载、进程启动、推理及输出读取墙钟耗时.
- `pipeline_wall_ms_excluding_output_write`: 图片读取、预处理、推理、后处理与提示选择耗时.
- `python_peak_rss_kib`: Python 进程峰值 RSS.
- `child_peak_rss_kib`: neuronrt 子进程峰值 RSS,不能与 Python 峰值简单相加.
- `neuronrt.log`: 单次硬件推理原始日志.
- `warmup.log` / `benchmark.log`: 各自独立进程执行 10 次和 100 次推理的原始日志.

预热进程与计时进程不同,不能声称计时阶段复用了同一 runtime 实例的热状态.
纯 NPU 延迟须从实际 runtime 日志确认口径后填写,不能用 CLI 墙钟耗时替代.
正式发布需同时记录板端系统、Runtime 版本、模型哈希、电源/性能档位与统计方法.
