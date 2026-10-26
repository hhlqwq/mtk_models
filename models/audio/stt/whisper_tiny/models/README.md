# 模型产物

原始权重 `tiny.pt` 与转换产物保存在 NAS 对应模型资源目录,本目录仅记录公开来源.

本目录预期生成以下产物:

- `tiny_encoder.onnx`：固定 `[1, 80, 3000]` Log-Mel 输入的 Encoder.
- `tiny_decoder_step.onnx`：单 Token、固定 200 Token KV Cache 的 Decoder.
- `encoder_fp32.tflite`、`decoder_step_fp32.tflite`：MTK Converter 产物.
- `encoder_fp32.dla`、`decoder_step_fp32.dla`：Genio 720 DLA.

实际文件、Shape、dtype 和编译参数必须以当前运行清单为准.
