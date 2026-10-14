# 模型资源

官方资源已下载并核验,[来源记录](source_url.txt) 与
[逐文件大小和 SHA-256](source_manifest.json) 随代码归档.

使用同一固定 revision 的 `model.safetensors`、`config.json`、
`generation_config.json`、`tokenizer.json`、`tokenizer_config.json` 和 `LICENSE`.
权重与生成产物不进入普通 Git 历史.

`generated/` 保存自行导出的 28 个 Decoder ONNX、10 个词表投影 ONNX,
对应 TFLite / FP16 DLA、FP16 embedding、评测输入与参考输出.
`artifact_manifest.json` 记录板端部署产物的大小与哈希,
板端加载前逐文件核对.本实现使用自有 Neuron Runtime 桥接库,
不是 `llm_cmdline_tool` 的 Prompt / Generative 包格式.
不要将地瓜或高通预编译模型复制到这里作为移植源.

模型文件已改为使用外部资源目录,详见上级 README 的模型资源路径说明.此目录只保留源码、文档和公开来源记录.
