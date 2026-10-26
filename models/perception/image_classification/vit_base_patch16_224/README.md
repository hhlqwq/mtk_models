# ViT-Base Patch16 224

> Python 推理代码已迁移,尚未做板端验证.下文报告和示例属于迁移前的 C++/原生 CLI 实现,不能作为当前 Python 实现的精度、性能或 RSS.

ImageNet-1K 图像分类，输入为 224×224 RGB 图像。正式上游为 PyTorch Vision v0.15.1，来源见[模型卡](model_card.md)。

量化方式: INT8 训练后量化 (PTQ),采用逐输出通道权重量化。

## 执行范围

先在已有编译容器执行 `bash deploy/compile.sh`,完成转换、量化、编译与上传后结束.随后登录开发板,在部署目录执行 `bash compile.sh`,做少量通路检查并保存 README 示例到 `examples/output/`.该阶段不计算或发布精度、性能和峰值 RSS 报告.

正式评价使用独立的 `evaluate.sh`: 容器端只计算已有浮点模型的同协议参考及必要评价输入,资源单独放入已有部署包后,板端运行 `bash evaluate.sh`,统计精度、性能和推理进程峰值 RSS.正式结果只写入 `results/summary.json`,不重新量化或编译.预测、日志及内部指标位于当天日期临时目录.详见[板端规范](../../../../docs/genio_720.md).

下文已归档结果保留原始模型版本和评价范围,本次入口调整不代表重新实测.

## 第一步: 编译并上传

模型权重、校准与验证数据、输出和板端目录在 [config.yaml](config.yaml) 配置.默认校准按文件名排序跳过前 1000 张,再取 100 张;模型与中间产物写入 NAS 资源目录.

编译主机的验证集图片目录支持 `val/` 或解压后的 `ILSVRC2012_img_val/`,根目录需有 `val_labels_0based.txt`。标签由官方 devkit 的 `meta.mat` 和验证集标签映射为 TorchVision 的 0–999 类别编号,不能直接将官方编号减 1。

在编译主机的本模型目录运行:

```bash
bash deploy/compile.sh
```

入口在当前容器中完成量化、DLA 编译、ONNX 全量 Top-1 评价和上传。Docker 需安装 ONNX Runtime、NumPy、OpenCV、PyTorch、torchvision 和 tqdm。

## 第二步: 独立评价

上传完成后,请登录开发板,在部署目录执行 `bash compile.sh` 完成通路检查和示例生成.正式评价前,按[板端规范](../../../../docs/genio_720.md)准备同协议参考及必要输入,在已部署目录执行:

```bash
bash evaluate.sh
```

精度、性能和峰值 RSS 统一保存到 results/summary.json,过程数据位于日期临时目录,不重新量化或编译.

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

板端通过 Python 完成图片预处理和 NPU 推理,计算 Top-1、NPU 平均耗时、推理进程峰值 RSS 与相对 ONNX 的精度变化。板端需安装 Python 3、NumPy 和 OpenCV。

评价结果只保存在部署目录的 `results/summary.json`.预测、日志和中间指标保留在 `/tmp/hailongcodex/<当天日期>/<模型>_evaluate_<运行编号>/`;展示示例保存在 `examples/output/`.取回唯一报告后据实更新 README.

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

compile.sh 复用固定少量样本的板端预测,生成展示文件到部署目录的 `examples/output/`,不依赖全量评价.
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
├── compile.sh
├── evaluate.sh          # 独立精度、性能与 RSS 评价.
├── config.json          # 不含连接地址的模型配置.
├── models/              # 模型与推理所需参数.
├── board/               # 板端程序、评测代码及必要依赖.
├── examples/            # 少量输入、来源清单和展示输出.
└── results/summary.json  # 唯一正式评价报告.
```

评价结果只保存在部署目录的 `results/summary.json`.预测、日志和中间指标保留在 `/tmp/hailongcodex/<当天日期>/<模型>_evaluate_<运行编号>/`;展示示例保存在 `examples/output/`.取回唯一报告后据实更新 README.
## 文件结构

- `deploy/compile.sh`: 编译部署与展示示例入口;精度、性能和 RSS 使用独立的 `deploy/evaluate.sh`.
- `deploy/host/`: 编译主机使用的导出、转换与辅助工具.
- `deploy/board/`: 板端程序源码、预处理和评测代码.
- `models/`: 原始权重、转换产物与来源说明; 必要的上游源码放在 `models/upstream/`.
- `examples/`: 少量固定输入与实际板端效果输出.
- `results/summary.json`: 上传后的最新测试汇总.

### 输出误差统计

`evaluate.sh` 同时计算 CosSim、RMSE 和 NRMSE,写入唯一 `results/summary.json` 的 `output_comparison`.浮点参考生成时保存 output_reference.sqlite,与任务参考一起放入板端 models/;按量化前输入内容配对,在现有 NPU 评价过程中逐输出累计统计.NRMSE 使用 `RMSE / RMS(浮点参考输出) × 100%`;多输出分别统计,缺参考如实记录缺失.具体定义、资源位置及当前未实测边界见[输出误差统计](../../../../docs/output_metrics.md).本次仅增加统计代码,未重新执行评价,历史指标不代表新增统计已验证.
