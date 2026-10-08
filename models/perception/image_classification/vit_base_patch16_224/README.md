# ViT-Base Patch16 224

ImageNet-1K 图像分类，输入为 224×224 RGB 图像。正式上游为 PyTorch Vision v0.15.1，来源见[模型卡](model_card.md)。

量化方式: INT8 训练后量化 (PTQ),采用逐输出通道权重量化。

## 第一步：编译并上传

在 [run.sh](deploy/run.sh) 顶部填写以下路径:

| 配置 | 内容 |
| --- | --- |
| `MODEL_ONNX` | 待量化 ONNX,留空使用 `models/model_mtk_compatible.onnx` |
| `ONNX_DATASET_DIR` | 编译主机上的 ImageNet 验证集根目录 |
| `CALIBRATION_DIR` | 校准图片目录,通常使用上述验证集的 `val/` |
| `BOARD_HOST` | 板端 SSH 用户和地址 |
| `BOARD_DATASET_DIR` | 板端 ImageNet 验证集根目录 |
| `BOARD_DEPLOY_DIR` | 板端模型与程序部署目录 |

模型、校准数据和 ONNX 评测数据必须同时对编译主机与 Docker 可见。模型产物写入 `MODEL_OUTPUT_DIR`,默认使用本模型的 `models/` 目录。当前校准从按文件名排序的第 1001 张图片开始使用 100 张图片。

在**编译主机的仓库根目录**运行:

```bash
bash models/perception/image_classification/vit_base_patch16_224/deploy/run.sh
```

脚本在 Docker 中量化并编译 DLA、评测 ONNX 全量 Top-1,在编译主机交叉编译板端 C++ 程序,然后上传模型、程序与实测 ONNX 基准。Docker 需安装 ONNX Runtime、NumPy、OpenCV 和 tqdm。

## 第二步：开发板测试

登录脚本中配置的 `BOARD_HOST`,执行第一步打印的板端命令。也可进入 `BOARD_DEPLOY_DIR` 对应的实际目录运行:

```bash
bash run.sh
```

ViT 默认执行 ImageNet 全量测试,无需额外参数。编译主机与板端均使用相同的 50000 张验证图片和零基类别标签:

```text
数据集根目录/
├── val/
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ...
│   └── ILSVRC2012_val_00050000.JPEG
└── val_labels_0based.txt
```

板端通过 C++ 完成图片预处理和 NPU 推理,计算 Top-1、NPU 平均耗时、推理进程峰值 RSS 与相对 ONNX 的精度变化。板端需安装 Python 3。

结果写入 `BOARD_RESULTS_DIR/<运行编号>/summary.json`,默认位于部署目录的 `results/`。成功后只保留汇总文件; 失败时保留现场。将结果上传到本模型的 `results/summary.json` 后更新 README。

## 当前测试结果

待上传本模型的 `results/summary.json` 后更新。只记录板端 NPU 平均耗时、推理进程峰值 RSS (MiB)、任务核心精度、同协议 ONNX 参考精度、部署精度和精度变化。峰值 RSS 包含运行库及前后处理,不代表 NPU 专用内存。

## 文件结构

- `deploy/run.sh`: 编译上传和板端测试的唯一 Shell 入口.
- `deploy/python/`: 模型导出、转换、评测与辅助代码.
- `deploy/cpp/`: 板端 C++ 源码.
- `models/`: 模型产物与来源说明.
- `examples/`: 示例输入与输出,按需保留.
- `results/summary.json`: 上传后的最新测试汇总.
