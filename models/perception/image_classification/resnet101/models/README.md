# ResNet101 资源

正式资源位于本地配置的 `models_root/perception/image_classification/resnet101/`,仓库只保留代码与来源记录.

请下载 [resnet101-cd907fc2.pth](https://download.pytorch.org/models/resnet101-cd907fc2.pth)到该目录.它对应 TorchVision `ResNet101_Weights.IMAGENET1K_V2`,约 170.5 MiB;不使用名称不同的 V1 权重或厂商导出产物替代.

编译入口从本地文件生成 `model_fp32.onnx`、`model_int8.dla`,`build/` 保存 INT8 TFLite 和 `quantization.json`.独立评价使用已有 ONNX,在 `build/onnx_reference.json` 生成完整参考,由用户单独放入板端部署包.不在模型资源代码目录存放权重或转换产物.

完整地址、用途与固定版本见 [source_url.txt](source_url.txt),权重未在本仓库内下载.
