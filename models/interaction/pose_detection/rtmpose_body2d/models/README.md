# 模型产物

正式产物必须从待锁定的 OpenMMLab MMPose 官方配置和权重自行导出.当前已有文件来自
历史 Qualcomm v0.61.0 ONNX 归档,只用于保留工程证据,不得作为新交付的原始模型或
正式指标基线.

本地生成产物不进入普通 Git:

- `model_fp32.onnx` 与 `rtmpose_body2d.data`: 历史 Qualcomm v0.61.0 FP32 ONNX 及外部权重.
- `model_mtk_compatible.onnx`: 合并外部权重,逐节点检查后将 opset 限制到 18、IR 限制
  到 8,并移除不受 MDLA 支持的 RGB 到 BGR 前缀;兼容模型直接接收 BGR,固定输入的
  GAU 双轴广播会等价展开为同形状乘法.固定输入的双输出数值等价验证通过后才允许转换.
- `model_int8.tflite`: MTK Converter 8.16.0 生成的 INT8 模型.
- `model_int8.dla`: NCC 8.2.31 使用 `mdla5.3`、`--suppress-output` 和
  `--disallow-bridge` 生成的 Genio 720 模型.
- `SHA256SUMS`: 上述归档及模型产物校验值.

固定输入为 `image [1,3,256,192]`,输出为 `pred_x [1,133,384]` 和
`pred_y [1,133,512]`;板端输出顺序以 TFLite tensor details 为准,不依赖节点名称猜测.
