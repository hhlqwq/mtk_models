# 模型产物
下载地址： https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/vit/releases/v0.61.0/vit-onnx-float.zip
预期产物为 `model_fp32.onnx`、`model_int8.tflite` 和 `model_int8.dla`。是否生成 FP16 ONNX
取决于 Genio 720 转换兼容性；不为满足目录名称而伪造无效 FP16 文件。
