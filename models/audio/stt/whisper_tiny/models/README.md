# 模型产物

原始权重 `tiny.pt` 与转换产物统一保存在本目录,不进入 Git.

本目录预期生成以下产物:

- `encoder_fp32.onnx`：固定 `[1, 80, 3000]` Log-Mel 输入的 Encoder.
- `decoder_step_fp32.onnx`：单 Token、固定 200 Token KV Cache 的 Decoder.
- `encoder_fp32.tflite`、`decoder_step_fp32.tflite`：MTK Converter 产物.
- `encoder_fp32.dla`、`decoder_step_fp32.dla`：Genio 720 DLA.

实际文件、Shape、dtype 和编译参数必须以当前运行清单为准.
