# ViT-Base Patch16 224

ImageNet-1K 图像分类，输入为 224×224 RGB 图像。正式上游为 PyTorch Vision v0.15.1，来源见[模型卡](model_card.md)。

量化方式: INT8 训练后量化 (PTQ),采用逐输出通道权重量化。

## 执行范围

`deploy/run.sh` 和板端 `run.sh` 无参数默认小批量验证: ImageNet 固定 200 张图片.评价范围不改变校准、量化或编译配方.

全量精度使用独立入口: 容器执行 `bash deploy/evaluate.sh` 生成已有浮点模型的全量参考,单独放入现有部署包后,板端执行 `bash evaluate.sh`.该流程不量化、不编译、不上传,仅推理和计算精度,结果保存为本次运行的 `accuracy_summary.json`.详见[板端评价规范](../../../../docs/genio_720.md).

下文已归档指标保留原运行范围,不因默认入口变化而更新.

## 第一步: 编译并上传

模型权重、校准与验证数据、输出和板端目录在 [config.yaml](config.yaml) 配置.默认校准按文件名排序跳过前 1000 张,再取 100 张;模型与中间产物写入 NAS 资源目录.

编译主机的验证集图片目录支持 `val/` 或解压后的 `ILSVRC2012_img_val/`,根目录需有 `val_labels_0based.txt`。标签由官方 devkit 的 `meta.mat` 和验证集标签映射为 TorchVision 的 0–999 类别编号,不能直接将官方编号减 1。

在编译主机的本模型目录运行:

```bash
bash deploy/run.sh
```

入口在当前容器中完成量化、DLA 编译、ONNX 全量 Top-1 评价、C++ 构建和上传。Docker 需安装 ONNX Runtime、NumPy、OpenCV、PyTorch、torchvision 和 tqdm。

## 第二步: 开发板测试

登录脚本中配置的 `BOARD_HOST`,执行第一步打印的板端命令。也可进入 `BOARD_DEPLOY_DIR` 对应的实际目录运行:

```bash
bash run.sh
```

## 数据与精度评测

ViT 默认验证固定 200 张图片.完整数据仍按下列 ImageNet 50000 张验证图片和零基标签格式准备,全量评价单独执行 `evaluate.sh`:

```text
数据集根目录/
├── val/ 或 ILSVRC2012_img_val/
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ...
│   └── ILSVRC2012_val_00050000.JPEG
└── val_labels_0based.txt
```

板端通过 C++ 完成图片预处理和 NPU 推理,计算 Top-1、NPU 平均耗时、推理进程峰值 RSS 与相对 ONNX 的精度变化。板端需安装 Python 3、NumPy 和 OpenCV。

结果写入 `BOARD_RESULTS_DIR/<运行编号>/summary.json`,默认位于部署目录的 `results/`。成功后保留汇总和少量效果示例; 失败时保留现场。将结果上传到本模型的 `results/summary.json` 后更新 README。

## 当前测试结果

结果来自 [results/summary.json](results/summary.json),运行编号 `20261008_094211_126146`,ImageNet val2012 全量 50000 张图片。

| 指标 | 本次结果 |
| --- | ---: |
| 板端 NPU 平均推理耗时 | **51.75 ms** |
| 板端推理进程峰值 RSS | **141.30 MiB** |
| ONNX FP32 Top-1 | **80.66%** |
| 板端 Top-1 | **79.40%** |
| 精度变化 (百分点) | **-1.27** |

ONNX 与板端使用同一份验证集和评测协议.精度以百分比显示并保留两位小数,变化按原始数值计算,负数表示下降.完整数值保存在 `summary.json` 中。

NPU 耗时为常驻模型对 50000 张图片的 `NeuronRuntime_inference` 平均调用耗时,不含图片读取、前后处理及精度计算.峰值 RSS 为推理进程峰值常驻内存,包含运行库与前后处理,不代表 NPU 专用内存。

## 效果示例

原图下方显示板端 Top-1 与 Top-5 类别排序.

输入已放在 `examples/input/`,选取 [ImageNet val2012](https://image-net.org/challenges/LSVRC/2012/) 的前三张图片: 海蛇、雪山和牧羊犬.来源与样本清单在编译时自动生成并随部署上传,不纳入 Git.

全量测试自动复用选定样本的板端预测,生成少量效果文件到本次结果目录的 `examples/output/`.
以下三张效果来自本次板端实际推理,复用全量测试样本的预测,未额外运行模型.

### 示例 1: 海蛇

![海蛇板端分类结果](examples/output/sample_1_classification.jpg)

### 示例 2: 雪山

![雪山板端分类结果](examples/output/sample_2_classification.jpg)

### 示例 3: 牧羊犬

![牧羊犬板端分类结果](examples/output/sample_3_classification.jpg)

## 板端部署结构

第一步上传到 `BOARD_DEPLOY_DIR` 后的布局统一为:

```text
部署目录/
├── run.sh
├── evaluate.sh          # 独立全量精度.
├── board_paths.conf
├── models/              # 模型与推理所需参数.
├── board/               # 板端程序、评测代码及必要依赖.
├── examples/input/      # 少量示例输入及来源清单.
└── results/             # 汇总及少量效果示例.
```

全量测试结果保存在 `BOARD_RESULTS_DIR/<运行编号>/summary.json`.成功后保留汇总和少量效果示例,失败时保留本次工作目录.将汇总上传为本模型的 `results/summary.json` 后更新 README.
## 文件结构

- `deploy/run.sh`: 部署和默认小批量验证入口;全量精度使用独立的 `deploy/evaluate.sh`.
- `deploy/host/`: 编译主机使用的导出、转换与辅助工具.
- `deploy/board/`: 板端程序源码、预处理和评测代码.
- `models/`: 原始权重、转换产物与来源说明; 必要的上游源码放在 `models/upstream/`.
- `examples/`: 少量固定输入与实际板端效果输出.
- `results/summary.json`: 上传后的最新测试汇总.
