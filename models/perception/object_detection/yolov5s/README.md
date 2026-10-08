# YOLOv5s · Genio 720

COCO 80 类目标检测。模型来源和许可证见[模型卡](model_card.md)。

量化方式: INT8 训练后量化 (PTQ),使用 100 张图片校准.

## 第一步: 编译并上传

所有配置均在脚本顶部,按模型与校准数据、产物与临时目录、板端地址与数据、ONNX 精度数据、编译环境分组.脚本已填写当前部署环境的路径,使用时按注释调整等号右侧的值; 编译主机和 Docker 须能访问相同数据,板端路径独立配置.

在 [run.sh](deploy/run.sh) 顶部确认 `MODEL_WEIGHTS`、`CALIBRATION_DIR`、`BOARD_DEPLOY_DIR` 和 `BOARD_DATASET_DIR`,按需调整 `MODEL_OUTPUT_DIR`、`OUTPUT_DLA`。其余配置仅在 Docker、SDK 或交叉编译环境变化时修改。资源地址见 [source_url.txt](models/source_url.txt)。模型、校准集及产物路径必须同时对编译主机和 Docker 容器可见。

在编译主机的本模型目录运行:

```bash
bash deploy/run.sh
```

脚本在 Docker 中生成 DLA,自动评测 ONNX FP32 全量精度,在编译主机交叉编译板端 C++ 程序,然后上传文件和实测 FP32 基准。`FP32_IMAGES_DIR` 与 `FP32_ANNOTATIONS` 必须指向与板端相同的 COCO val2017 5000 张图片及标注。校准仅使用其中 100 张图片,与全量精度评测分开。

浮点模型只记录 mAP@0.5:0.95,不测试 PyTorch 精度,不统计编译主机耗时或内存。Docker 需安装 ONNX Runtime、pycocotools、OpenCV、PyTorch、torchvision 和 tqdm; PyTorch 仅用于解码和 NMS。模型产物写入脚本配置的输出目录。脚本不下载模型或数据。

## 第二步: 开发板测试

使用脚本中配置的 `BOARD_HOST` 登录开发板，并执行第一步打印的 `[NEXT]` 命令。

板端使用 C++ 完成 JPEG 预处理、Neuron Runtime 推理、YOLO 解码和 NMS,直接生成 COCO 预测与耗时记录,无需回传 5000 张图片的原始 NPU 输出。

进入实际配置的部署目录后运行:

```bash
bash run.sh
```

默认执行 COCO val2017 全量 5000 张图片测试,无需参数.

## 数据与精度评测

结果写入 `BOARD_RESULTS_DIR`，默认位于 `BOARD_DEPLOY_DIR/results`。测试要求 `BOARD_DATASET_DIR/images/` 中有 5000 张 COCO val2017 图片、`BOARD_DATASET_DIR/annotations/instances_val2017.json`，板端还需 `pycocotools`。若 SSH 提示主机密钥变化，先核对板端指纹，再更新编译主机上执行脚本用户的 `known_hosts`。

## 当前测试结果

结果来自 [results/summary.json](results/summary.json),运行编号 `20261008_032807_67603`,COCO val2017 全量 5000 张图片。

| 指标 | 本次结果 |
| --- | ---: |
| 板端 NPU 平均推理耗时 | **9.76 ms** |
| 板端推理进程峰值 RSS | **33.23 MiB** |
| ONNX FP32 mAP@0.5:0.95 | **37.09%** |
| 板端 mAP@0.5:0.95 | **35.86%** |
| 精度变化 (百分点) | **-1.23** |

ONNX FP32 与板端精度均来自本次同协议 COCO val2017 全量实测。mAP 以百分比显示并保留两位小数,精度变化按原始数值计算,负数表示下降。`summary.json` 保留完整数值精度,已纳入 Git 管理。

NPU 耗时统计预热后 5000 张图片的 `NeuronRuntime_inference` 调用,不含图片读取、前后处理和 COCOeval。峰值内存为板端 C++ 推理进程峰值 RSS,包含运行库与前后处理,不代表 NPU 专用内存。

全量测试成功后只保留一个 `summary.json`,中间预测和日志自动删除; 失败时保留现场。测试完成后将汇总文件上传到本地 `results/summary.json`,据此更新当前结果。

## 示例输出

三张示例图仅展示检测效果,不参与 COCO 精度评测。

### 示例 1

![示例 1 检测结果](examples/output/sample_1_detections.jpg)

### 示例 2

![示例 2 检测结果](examples/output/sample_2_detections.jpg)

### 示例 3

![示例 3 检测结果](examples/output/sample_3_detections.jpg)

示例输入由项目维护者使用 OpenAI 图像生成工具制作,按 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) 发布。

## 板端部署结构

第一步上传到 `BOARD_DEPLOY_DIR` 后的布局统一为:

```text
部署目录/
├── run.sh
├── board_paths.conf
├── models/              # 模型与推理所需参数.
├── board/               # 板端程序、评测代码及必要依赖.
└── results/             # 测试结果.
```

全量测试结果保存在 `BOARD_RESULTS_DIR/<运行编号>/summary.json`.成功后只保留汇总文件,失败时保留本次工作目录.将汇总上传为本模型的 `results/summary.json` 后更新 README.
## 文件结构

- `deploy/run.sh`: 编译上传和板端测试的唯一 Shell 入口.
- `deploy/host/`: 编译主机使用的导出、转换与辅助工具.
- `deploy/board/`: 板端程序源码、预处理和评测代码.
- `models/`: 原始权重、转换产物与来源说明; 必要的上游源码放在 `models/upstream/`.
- `examples/`: 示例输入与输出,按需保留.
- `results/summary.json`: 上传后的最新测试汇总.
