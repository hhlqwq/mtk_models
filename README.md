# MTK Models

目标编译环境为 Ubuntu 22.04 预装镜像, MTK SDK、Python 3.11 和 CUDA Torch 在构建阶段安装.
Ubuntu 系统依赖使用基础镜像官方软件源, pip 镜像地址在容器内永久配置.
Python 3.11.11 源码地址由 Docker 构建参数管理, 默认使用服务器已验证可达的镜像地址.
2026-09-07 已将同名容器迁移到 Ubuntu 22.04 镜像,并删除旧 Debian 12 容器和旧镜像.
创建容器时仅验证工具, 操作方法见 [Docker 说明](docker/README.md).

本项目面向 MediaTek Genio 720（MT8189）和 Genio 5100,对开源模型进行兼容性修改、
转换、量化、部署与板端验证.模型交付结构、文档完整度和结果展示方式参考
[Qualcomm AI Hub Models](https://huggingface.co/qualcomm/models),但不使用 Qualcomm
模型或其预导出产物作为 MTK 模型的移植源.每个模型都应形成"开源上游、原始框架、
标准格式导出、MTK NPU 部署、Demo、性能/精度报告、文档"的完整闭环.

## 模型来源原则

- `source` 只记录模型作者或官方开源项目发布的实现与权重,并记录版本、下载地址和许可证.
- `delivery_reference` 仅记录目录组织、文档和结果展示的参考页面,不得作为模型输入产物.
- 禁止以 Qualcomm 预导出的 ONNX、QNN、DLC 或其他转换产物作为正式移植起点.

## 首批模型范围

首批范围是图片"Target AI models"列出的全部模型,覆盖 Perception、Interaction、Navigation、
Gen AI、Audio、SLAM、PointCloud 和 3D 八个 Scenario.完整机器可读清单见
[target_models.yaml](registry/target_models.yaml).

YOLOv5、ViT 和 RTMPose 是首批模型中的三个先行实现,用于率先打通目录、转换、MTK NPU
部署、Demo、精度和性能报告闭环.

部分模型提供公开样例及板端结果,便于展示推理输出.公开样例不替代正式数据集精度.

## 先行实现模型

| 模型 | 任务 | 标准输入 | 来源 | 当前状态 |
| --- | --- | --- | --- | --- |
| [Depth Anything V2 Small](models/navigation/single_camera_depth/depth_anything_v2_small/README.md) | Navigation / Single camera depth | 518×518 RGB | DepthAnything 官方源码与 Small 权重 | 待上传当前测试汇总 |
| [FastSAM-s](models/navigation/segmentation/fastsam/README.md) | Navigation / Segmentation | 640×640 RGB | CASIA-LMC-Lab / Ultralytics 8.0.111 | 待上传当前测试汇总 |
| [YOLOv5s](models/perception/object_detection/yolov5s/README.md) | Perception / Object detection | 640×640 RGB | Ultralytics | 本次全量结果已上传 |
| [ViT-Base Patch16 224](models/perception/image_classification/vit_base_patch16_224/README.md) | Perception / Image classification | 224×224 RGB | PyTorch Vision v0.15.1 | 待上传当前测试汇总 |
| [RTMPose Body2d](models/interaction/pose_detection/rtmpose_body2d/README.md) | Interaction / Pose detection | 256×192 RGB | OpenMMLab MMPose v1.3.2 | 待上传当前测试汇总 |
| [MobileFaceNet](models/interaction/face_recognition/mobilefacenet/README.md) | Interaction / Face recognition | 112×112 RGB 对齐人脸 | foamliu/MobileFaceNet | 待上传当前测试汇总 |
| [YOLO-World XL](models/perception/object_detection/yoloworld_xl/README.md) | Perception / Object detection | 640×640 RGB | AILab-CVC / MediaTek Model Zoo ONNX | 待上传当前测试汇总 |
| [Whisper-Tiny](models/audio/stt/whisper_tiny/README.md) | Audio / STT | 16 kHz 单声道、最长 30 秒 | OpenAI Whisper v20250625 | 待上传当前测试汇总 |

公开示例结果见各模型 README; 当前指标仅依据用户上传的 `results/summary.json` 更新.

各模型的当前数据集、精度和耗时见 README,机器可读结果统一保存为 `results/summary.json`。

状态只能使用以下四类：

- `环境建设中`：目录和工具已准备,尚未完成转换.
- `转换完成`：已生成 MTK 可编译模型,但尚未完成板端验证.
- `板端已验证`：模型已在目标板运行,但精度或性能报告仍不完整.
- `完整交付`：转换、Demo、板端性能、精度和文档全部完成.

## 开发与测试环境

| 项目 | 配置 |
| --- | --- |
| 编译主机 | Ubuntu 24.04；连接地址由用户配置 |
| 仓库目录 | 用户在编译主机上放置本仓库的位置 |
| Docker 镜像 | `openexplorer/ai_toolchain_ubuntu_22_g720_gpu:np8.0.11`；镜像 ID `9ac9238a70ec` |
| 当前 Docker 容器 | `hhl_g720_8011`；已完成 ViT 与 RTMPose 转换和板端验证 |
| Python | 当前容器 3.11.11 |
| NeuroPilot SDK | 8.0.11 |
| MTK Converter | 8.16.0 |
| Neuron Compiler | 8.2.31 |
| Genio 720 EVK | 连接地址由用户在模型脚本中配置 |
| 板端目录 | 模型、数据集和结果位置由用户在模型脚本中配置 |
| 板端系统 | Rity Demo 26.0-release / Scarthgap / Linux 6.6.137 |
| 板端 Neuron Runtime | 8.2.16 |
| 板端 ONNX Runtime | 1.20.2；包含 Neuron、XNNPACK 和 CPU Execution Provider |
| 板端 GAI 工具 | `/usr/sbin/llm_cmdline_tool`；工具存在已验证，LLM/VLM 模型推理待验证 |

编译主机和 Docker 容器必须能以相同绝对路径访问模型输入与输出目录。

详细说明见 [环境文档](docs/environment.md)、[Genio 720 板端规范](docs/genio_720.md) 和
[2026-09-07 官网与实际环境核对记录](docs/genio_720_environment_audit_20260907.md).

2026-09-21 已通过 Windows 主机和 `USB 3.2 P0` 将官方 eMMC v26.0 镜像完整刷入
Genio 720 EVK.当前平台基线和刷写证据见
[Genio 720 v26.0 升级记录](docs/genio_720_v26_upgrade_20260921.md).现有模型报告中的
`26.0-dev / 6.6.117` 是对应历史运行的真实环境,不会批量改写；这些模型在正式 v26.0
上的兼容性、精度和性能需要重新执行后才能更新状态.

## 快速开始

在编译主机准备 Docker 环境：

```bash
source env.sh
bash ./docker/create_container.sh
bash ./docker/enter_container.sh
```

当前同名容器已绑定目标 Ubuntu 22.04 镜像.旧容器与旧镜像已在留存迁移证据后删除；
迁移备份属于历史环境记录，具体位置由环境管理员保存。

模型权重和数据集等大文件由用户准备并放置到编译主机可访问的位置；
编译主机和开发板不在运行脚本时自动下载模型、数据集或补丁。允许在本机工作区下载并
纳入 Git 的小型源码包、补丁和配置文件.服务器上的模型准备脚本只允许执行离线校验、解压
和转换,不允许包含 `curl`、`wget`、`git clone` 或 Hugging Face 在线下载.

进入容器后,根据具体模型 README 执行离线准备、转换、编译和部署脚本.正式评测数据集不会
自动下载,也不会将缺少标注的样例推理结果写成正式精度.

## 模型目录规范

```text
mtk_models/
├── configs/                  # 平台与工具链版本.
├── docker/                   # 可复现的 Ubuntu 转换环境.
├── registry/                 # 上百模型的统一注册表与状态定义.
├── models/
│   └── scenario_name/
│       └── category_name/
│           └── model_name/   # 单模型完整交付目录.
├── templates/model/          # 新模型标准模板.
├── tools/                    # 公共下载、转换、部署、评测工具.
├── reports/                  # 按目标平台汇总的报告.
└── docs/
```

单模型目录保持完整闭环：

```text
models/scenario_name/category_name/model_name/
├── README.md
├── model_card.md
├── LICENSE
├── original/
│   └── source_url.txt
├── models/
│   ├── model_fp32.onnx
│   ├── model_fp16.onnx
│   ├── model_int8.tflite
│   └── model_int8.dla
├── deploy/
│   ├── run.sh               # 编译主机与开发板共用的入口.
│   └── inference_demo/
├── examples/
│   ├── input/
│   └── output/
└── results/
    └── summary.json
```

每个模型使用一个脚本完成两步流程：在编译主机运行 `deploy/run.sh` 编译并上传，再在开发板运行上传的同一脚本；路径在脚本中配置。具体命令见各模型 README。

模型产物与板端结果分别写入用户配置的 MODEL_OUTPUT_DIR 和 BOARD_RESULTS_DIR。

完整板端测试成功后只保留一个 `summary.json`,包含平均 NPU 耗时、核心精度、匹配参考基准及精度差值。测试完成后将文件上传到本地模型的 `results/summary.json`,据此更新 README; 未上传结果不预填数字.



仓库通过 `registry/models.yaml` 维护模型索引,避免扫描上百个目录才能了解交付状态.

大模型文件、转换产物、输入数据和输出数据默认不进入普通 Git 历史.正式发布模型文件时应使用
Git LFS 或 Release。用户放置的官方开源权重、历史
Qualcomm 对照资产和标签副本由 `.gitignore` 排除并保留在各自工作环境中.

## 验收原则

每个模型必须同时满足以下条件才可标记为"完整交付"：

1. 记录官方开源项目、源码版本、权重版本、许可证和下载地址.
2. 从开源上游权重运行原始框架推理并保存可复现命令.
3. 自行导出 ONNX 或 TFLite,并与原始框架完成数值或任务指标对比.
4. 通过 MTK Converter 和 Neuron Compiler 生成板端模型.
5. 在 Genio 720 EVK 的 NPU 上完成 Demo 推理.
6. 分别报告纯 NPU 延迟、端到端延迟和峰值内存.
7. 使用正式标注数据集对比原始模型、ONNX 和 MTK NPU 精度.
8. README、模型卡、精度报告和性能报告没有占位值.

## 新增模型

使用统一模板创建模型目录：

```bash
python tools/create_model.py \
    --scenario perception \
    --category image_classification \
    --model-id resnet50 \
    --model-name "ResNet-50"
```

创建后还必须将模型加入 `registry/models.yaml`,并填写真实来源、输入输出和目标平台状态.
图片中的完整候选清单维护在 `registry/target_models.yaml`,未开始的模型不提前生成空交付目录.
可使用以下命令执行只读结构检查：

```bash
python tools/check_registry.py
```

各模型统一在编译主机评测 ONNX 核心精度,板端记录核心精度、推理耗时、峰值 RSS和相对 ONNX 的精度变化。量化方式、数据要求与测试步骤见各模型 README。
