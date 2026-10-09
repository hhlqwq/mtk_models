# 模型资源

原始 Google 权重 `yamnet.h5`、固定源码 `upstream/`、导出 ONNX、FP16/INT8 TFLite、
MDLA 5.3 DLA 和运行参数保存在本目录,不纳入普通 Git.
完整下载地址、许可证、大小及权重哈希见 [source_url.txt](source_url.txt).
`resource_manifest.json` 记录各下载文件 SHA256,`quantization.json` 记录真实 IO
精度、IO 配置及可选 INT8 校准清单.默认 FP16 不使用校准数据.
模型资源离线准备后运行 `deploy/run.sh`,通过 `YAMNET_PRECISION=int8` 切换对照精度.
