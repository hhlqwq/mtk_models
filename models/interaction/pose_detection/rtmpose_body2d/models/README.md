# 模型产物

本地生成产物不进入普通 Git:

- `model_fp32.onnx` 与 `rtmpose_body2d.data`: Qualcomm v0.61.0 FP32 ONNX 及外部权重.
- `model_int8.tflite`: MTK Converter 8.16.0 生成的 INT8 模型.
- `model_int8.dla`: NCC 8.2.31 使用 `mdla5.3`、`--suppress-output` 和
  `--disallow-bridge` 生成的 Genio 720 模型.
- `SHA256SUMS`: 上述归档及模型产物校验值.

固定输入为 `image [1,3,256,192]`,输出为 `pred_x [1,133,384]` 和
`pred_y [1,133,512]`;板端输出顺序以 TFLite tensor details 为准,不依赖节点名称猜测.
