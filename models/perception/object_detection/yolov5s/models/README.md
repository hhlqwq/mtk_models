# 模型产物

转换后应包含 `yolov5s.pt`、`yolov5s.torchscript`、`model_fp32.onnx`、
`model_int8.tflite` 和 `model_int8.dla`.这些二进制文件默认被 `.gitignore` 排除；正式发布时
使用 Git LFS 或 Release.

模型文件已改为使用外部资源目录,详见上级 README 的模型资源路径说明.此目录只保留源码、文档和公开来源记录.
