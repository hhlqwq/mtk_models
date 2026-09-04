# 环境说明

## 版本基线

服务器 `/data/users/hailong.he/data/MTKG720` 已包含以下官方工具：

| 工具 | 版本 | 用途 |
| --- | --- | --- |
| NeuroPilot SDK Basic | 8.0.11 build 20260211 | 主机端工具集合 |
| mtk_converter | 8.16.0 cp311 | PyTorch/ONNX 到 TFLite 转换和 PTQ |
| mtk_quantization | 8.2.1 | 量化工具 |
| ncc-tflite | Neuron 8.2.31 | TFLite 到 DLA 编译 |
| neuronrt | 板端 8.2.16 | DLA NPU 推理 |

主机编译器与板端运行时版本不同。每份 benchmark 都必须同时记录二者，避免只写模糊的
“NeuroPilot SDK 8”。

## Python 选择

项目使用 Python 3.11。依据是当前 SDK 自带 CPython 3.11 wheel，且 MTK 新版示例使用
Python 3.11 虚拟环境。旧 YOLOv5 文档中的 Python 3.7 是历史转换链，不作为本项目通用环境。

## 数据集目录与原则

89 服务器数据集根目录固定为：

```text
/data/users/hailong.he/nas_smb/Datasets/open_source/raw
```

容器内以相同绝对路径只读挂载。模型脚本和报告统一记录真实 NAS 路径，不使用 `/datasets` 别名。

- 数据集以相同绝对路径只读挂载到容器。
- 校准子集与正式精度评测集必须分开记录。
- 不允许将 100 张校准图片的指标标记成 COCO 正式 mAP。
- 不允许用单张分类样例推理结果代替 ImageNet Top-1。
