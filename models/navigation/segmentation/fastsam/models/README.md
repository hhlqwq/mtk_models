# FastSAM 模型产物

此目录只存放由官方权重自行转换的本地输出,模型二进制不进入普通 Git.

| 文件 | 内容 |
| --- | --- |
| model_fp32.onnx | opset 13,静态 640,10 个原始输出 |
| pytorch_reference.npz | 同一真实图片的 FP32 输入及原始输出 |
| export_manifest.json | 权重、源码实现、图片、模型哈希及导出检查结果 |
| model_int8.tflite | MTK INT8 校准产物 |
| model_int8.json | 量化参数、输出索引语义映射、校准图片清单 |
| runtime_config.csv | C++ 板端程序使用的输入输出张量契约 |
| model_int8.dla | MDLA 5.3 无桥接编译产物 |
| SHA256SUMS / DLA_SHA256SUMS | 产物完整性校验 |
| deployment_manifest.json | DLA、量化参数、ONNX 和 PyTorch 基线的构建绑定 |

2026-09-23 已从本次用户提供的官方权重生成 ONNX、INT8 TFLite 和 Genio 720 DLA.
本次仅使用 16 张校准图进行冒烟转换,产物和哈希保存在 89 项目目录,
尚未按正式精度协议发布模型文件.
