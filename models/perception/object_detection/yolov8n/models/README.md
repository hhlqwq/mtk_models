# 模型资源

离线放置官方 `yolov8n.pt`,具体来源、大小和哈希见 `source_url.txt`.
编译脚本生成 `model_fp32.onnx`,`model_int8.tflite`,`model_int8.dla`,
`runtime_config.csv` 和元数据.权重、转换产物和临时数据不进入普通 Git.

模型文件已改为使用外部资源目录,详见上级 README 的模型资源路径说明.此目录只保留源码、文档和公开来源记录.
