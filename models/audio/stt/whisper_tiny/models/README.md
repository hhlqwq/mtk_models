# 模型产物

本目录预期生成以下非 Git 产物：

- `encoder_fp32.onnx`：固定 `[1, 80, 3000]` Log-Mel 输入的 Encoder.
- `decoder_step_fp32.onnx`：单 Token、固定 200 Token KV Cache 的 Decoder.
- `encoder_fp32.tflite`、`decoder_step_fp32.tflite`：MTK Converter 产物.
- `encoder_fp32.dla`、`decoder_step_fp32.dla`：Genio 720 DLA.
- `SHA256SUMS`：权重及各级产物哈希.

实际文件、Shape、dtype 和编译参数必须以当前运行清单为准.
