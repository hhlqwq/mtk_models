# 模型产物

正式产物必须从待锁定的官方开源权重自行导出.当前已有文件来自历史 Qualcomm
v0.61.0 ONNX 归档,只用于保留工程证据,不得作为新交付的原始模型或正式指标基线.

历史归档地址： https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/vit/releases/v0.61.0/vit-onnx-float.zip
历史归档 SHA-256：`72b7d02dd5c3d1e09c59196ba14364549aac9e8ed1f8212ea5f5c78f7424632f`.

新链路预期产物为原始框架权重、自行导出的 `model_fp32.onnx`、转换中间件
`model_mtk_compatible.onnx`、
`model_int8.tflite` 和 `model_int8.dla`.是否生成 FP16 ONNX
取决于 Genio 720 转换兼容性；不为满足目录名称而伪造无效 FP16 文件.
