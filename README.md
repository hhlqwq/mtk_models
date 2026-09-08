# MTK Models

目标编译环境为 Ubuntu 22.04 预装镜像, MTK SDK、Python 3.11 和 CUDA Torch 在构建阶段安装.
Ubuntu 系统依赖使用基础镜像官方软件源, pip 镜像地址在容器内永久配置.
Python 3.11.11 源码地址由 Docker 构建参数管理, 默认使用服务器已验证可达的镜像地址.
2026-09-07 已将同名容器迁移到 Ubuntu 22.04 镜像,并删除旧 Debian 12 容器和旧镜像.
创建容器时仅验证工具, 操作方法见 [Docker 说明](docker/README.md).
YOLOv5s 正式板端精度路径使用 C++ 完成 JPEG 预处理、Neuron Runtime 推理、YOLO 解码和
NMS,直接在开发板生成 COCO 预测与耗时证据,不回传 5000 张原始 NPU 输出.

本项目面向 MediaTek Genio 720（MT8189）和 Genio 5100,建立与
[Qualcomm AI Hub Models](https://huggingface.co/qualcomm/models) 类似的模型交付仓库.
每个模型都应形成“来源、转换、NPU 部署、Demo、性能/精度报告、文档"的完整闭环.

## 首批模型范围

首批范围是图片“Target AI models"列出的全部模型,覆盖 Perception、Interaction、Navigation、
Gen AI、Audio、SLAM、PointCloud 和 3D 八个 Scenario.完整机器可读清单见
[target_models.yaml](registry/target_models.yaml).

YOLOv5、ViT 和 RTMPose 是首批模型中的三个先行实现,用于率先打通目录、转换、MTK NPU
部署、Demo、精度和性能报告闭环.

## 先行实现模型

| 模型 | 任务 | 标准输入 | 来源 | 当前状态 |
| --- | --- | --- | --- | --- |
| [YOLOv5s](models/perception/object_detection/yolov5s/README.md) | Perception / Object detection | 640×640 RGB | Qualcomm / Ultralytics | 完整交付 |
| [ViT-Base Patch16 224](models/perception/image_classification/vit_base_patch16_224/README.md) | Perception / Image classification | 224×224 RGB | Qualcomm | 板端已验证（绝对 Top-1 待标签） |
| [RTMPose Body2d](models/interaction/pose_detection/rtmpose_body2d/README.md) | Interaction / Pose detection | 256×192 RGB | Qualcomm | 环境建设中 |

状态只能使用以下四类：

- `环境建设中`：目录和工具已准备,尚未完成转换.
- `转换完成`：已生成 MTK 可编译模型,但尚未完成板端验证.
- `板端已验证`：模型已在目标板运行,但精度或性能报告仍不完整.
- `完整交付`：转换、Demo、板端性能、精度和文档全部完成.

## 开发与测试环境

| 项目 | 配置 |
| --- | --- |
| Ubuntu 编译服务器 | `ssh ubuntu89`；宿主 Ubuntu 24.04.4 LTS |
| 服务器工作目录 | `/data/users/hailong.he/github/mtk_models` |
| Docker 镜像 | `hhl_g720_311:ubuntu22.04-np8.0.11`；镜像 ID `006a427a61fd` |
| 当前 Docker 容器 | `hhl_g720_311`；Ubuntu 22.04.5 LTS,已完成迁移与运行验证 |
| Python | 当前容器 3.11.11 |
| NeuroPilot SDK | 8.0.11 |
| MTK Converter | 8.16.0 |
| Neuron Compiler | 8.2.31 |
| Genio 720 EVK | `root@192.168.0.92`,无密码 |
| 板端目录 | `/root/hailong.he` |
| 板端系统 | Rity Demo 26.0-dev / Scarthgap / Linux 6.6.117 |
| 板端 Neuron Runtime | 8.2.16 |

详细说明见 [环境文档](docs/environment.md)、[Genio 720 板端规范](docs/genio_720.md) 和
[2026-09-07 官网与实际环境核对记录](docs/genio_720_environment_audit_20260907.md).

## 快速开始

所有编译和转换必须在 89 Ubuntu 服务器执行：

```bash
ssh ubuntu89
cd /data/users/hailong.he/github/mtk_models
source env.sh
bash ./docker/create_container.sh
bash ./docker/enter_container.sh
```

当前同名容器已绑定目标 Ubuntu 22.04 镜像.旧容器与旧镜像已在留存迁移证据后删除；
迁移备份位于 89 的 `/data/users/hailong.he/data/MTKG720/migration_20260907/`.

模型权重和数据集等大文件必须由用户下载到本机工作区,再由用户同步或放置到 89 服务器；
89 服务器和 92 开发板禁止直接下载模型、数据集或补丁.允许由 Codex 在本机工作区下载并
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
├── model.yaml
├── LICENSE
├── original/
│   └── source_url.txt
├── models/
│   ├── model_fp32.onnx
│   ├── model_fp16.onnx
│   ├── model_int8.tflite
│   └── model_int8.dla
├── deploy/
│   ├── download_original.sh  # 仅离线校验和展开,不执行网络下载.
│   ├── convert.sh
│   ├── build.sh
│   ├── deploy_board.sh
│   └── inference_demo/
├── examples/
│   ├── input/
│   └── output/
└── docs/
    ├── accuracy.md
    └── benchmark.md
```

仓库通过 `registry/models.yaml` 维护模型索引,避免扫描上百个目录才能了解交付状态.

大模型文件、转换产物、输入数据和输出数据默认不进入普通 Git 历史.正式发布模型文件时应使用
Git LFS 或 Release,并在 `model_card.md` 中记录 SHA-256.

## 验收原则

每个模型必须同时满足以下条件才可标记为“完整交付"：

1. 锁定 Hugging Face 来源、版本、许可证和 SHA-256.
2. 原始模型推理成功并保存可复现命令.
3. ONNX 推理结果与原始模型完成数值或任务指标对比.
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

## 历史工程迁移

旧工程 `D:\code\gitee\mtk` 仅作为只读迁移来源.已有 YOLOv5n/YOLOv8s 转换经验、板端
推理命令及精度评估方法将逐项迁入当前结构,迁移记录见
[migration_from_gitee.md](docs/migration_from_gitee.md).
