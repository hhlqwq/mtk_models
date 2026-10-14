# 模型产物

从 PyTorch Vision v0.15.1 官方权重 `vit_b_16-c867db91.pth` 导出 ONNX,下载地址见 [source_url.txt](source_url.txt).

- `model_fp32.onnx`: 原始浮点模型.
- `model_mtk_compatible.onnx`: 用于 MTK 转换的兼容模型.
- `model_int8.tflite`、`model_int8.dla`: 量化与编译产物.
- `quantization.json`: 板端推理使用的量化参数.

产物位置在 `deploy/run.sh` 顶部配置.

模型文件已改为使用外部资源目录,详见上级 README 的模型资源路径说明.此目录只保留源码、文档和公开来源记录.
