# 性能报告

## 官方参考

MediaTek IoT AI Hub Model Zoo 记录的输入为 `3×640×640`、Float32：

| 平台 | 执行后端 | 官方纯模型延迟 |
| --- | --- | ---: |
| Genio 720 | Neuron EP | 403.15 ms |
| Genio 720 | CPU EP | 11214.33 ms |

官方数据由 `onnxruntime_perf_test` 测量，不能作为本项目实测结果。

## 本项目实测

板端兼容模型尚未完成成功推理。原始 opset 11 模型在 Neuron EP 建图时失败，故障证据为
`SoftMaxOpBuilder ... SinceVersion() < 13`，随后出现 `unregistered target: NEON`。
该失败不会被记录为延迟或 NPU 验证。

兼容模型运行后，本报告将填写会话创建耗时、预热次数、纯推理 mean/min/max/P50/P90/P95、
预处理、后处理、端到端耗时、峰值 RSS、Neuron/CPU 节点归属和模型 SHA-256。
