# 模型产物
下载地址： https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/vit/releases/v0.61.0/vit-onnx-float.zip
归档 SHA-256：`72b7d02dd5c3d1e09c59196ba14364549aac9e8ed1f8212ea5f5c78f7424632f`。
预期产物为原始基线 `model_fp32.onnx`、转换中间件 `model_mtk_compatible.onnx`、
`model_int8.tflite` 和 `model_int8.dla`。是否生成 FP16 ONNX
取决于 Genio 720 转换兼容性；不为满足目录名称而伪造无效 FP16 文件。
