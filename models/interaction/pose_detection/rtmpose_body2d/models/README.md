# 模型产物

正式产物从 OpenMMLab MMPose v1.3.2 官方配置和权重自行导出.当前旧 ONNX 文件来自
历史 Qualcomm v0.61.0 ONNX 归档,只用于保留工程证据,不得作为新交付的原始模型或
正式指标基线.

本地生成产物不进入普通 Git:

- `rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth`:
  官方 PyTorch 权重,完整 SHA-256 和 MD5 见 `model_card.md`.
- `model_fp32.onnx`: 从 MMPose v1.3.2 官方 PyTorch 权重自行导出的 FP32 基线.
- `model_mtk_compatible.onnx`: 逐节点检查后将 opset 限制到 18、IR 限制到 8,并在唯一
  图结构匹配时展开 NCC 不支持的 GAU 双轴广播.固定输入的双输出数值等价验证通过后
  才允许转换.
- `model_int8.tflite`: MTK Converter 8.16.0 生成的 INT8 模型.
- `model_int8.dla`: NCC 8.2.31 使用 `mdla5.3`、`--suppress-output` 和
  `--disallow-bridge` 生成的 Genio 720 模型.
- `SHA256SUMS`: 官方权重及自行生成模型产物的校验值.

固定输入为 NCHW RGB `[0,255]` 的 `image [1,3,256,192]`,输出为
`pred_x [1,133,384]` 和
`pred_y [1,133,512]`;板端输出顺序以 TFLite tensor details 为准,不依赖节点名称猜测.
