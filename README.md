# MTK Models

目标编译环境为 Ubuntu 22.04 预装镜像, MTK SDK、Python 3.11 和 CUDA Torch 在构建阶段安装.
Ubuntu 系统依赖使用基础镜像官方软件源, pip 镜像地址在容器内永久配置.
Python 3.11.11 源码地址由 Docker 构建参数管理, 默认使用服务器已验证可达的镜像地址.
创建容器时检查配置和挂载, 操作方法见 [Docker 说明](docker/README.md).

执行模型编译流程前,先从仓库根目录运行 `bash docker/enter_container.sh`,再在容器内执行各模型原有入口或独立 Python 步骤.模型入口只编排步骤,不再负责进入 Docker.编译、交叉编译和上传都在当前容器执行,板端评测仍在开发板执行.

Python 导入按标准库、第三方依赖、项目模块分组放在文件顶部,去除重复导入.多模型或多后端共用工具确有环境差异时,允许在对应分支加载可选依赖并注明原因; 从命令行指定并校验的上游源码须在确认路径后加载.不为兼容容器外执行而把普通导入散落在函数中.

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

本次补充模型为 [DeepSeek-R1-Distill-Qwen-1.5B](models/gen_ai/llm/deepseek_r1_distill_qwen_1_5b/README.md),
用于中英文文本问答与推理.使用 DeepSeek 官方权重,参考地瓜 RDK S 系列的部署方式;
原生 GAI W4A16 Prompt/Decode 交付包已完成板端双语推理,Decode 约 16.75 Token/s.
2026-10-10 已完成 Genio 720 FP16 双语 Demo 与样例数值检查 (`board_verified`),
Decode 约 3.8 Token/s,中文 / 英文分别在 73 / 412 Token 结束.Genio 5100 为 `not_started`;
W4A16 原生包已归档至 NAS,英文语法存在问题,正式量化质量待评测.
原始图片对应的 `target_models.yaml` 清单保持原样,补充模型登记于 `registry/models.yaml`.

| 模型 | 任务 | 标准输入 | 来源 | 当前状态 |
| --- | --- | --- | --- | --- |
| [DeepSeek-R1-Distill-Qwen-1.5B](models/gen_ai/llm/deepseek_r1_distill_qwen_1_5b/README.md) | Gen AI / LLM | Token + KV Cache,上下文 1024 | DeepSeek 官方权重 | Genio 720 W4A16 原生双语 Demo 通过,Decode 约 16.75 Token/s; 正式质量待评测 |
| [YAMNet](models/audio/ambient_sound/yamnet/README.md) | Audio / Ambient sound | 16 kHz 单声道,96×64 Log-Mel | Google TensorFlow Models 官方源码与权重 | Genio 720 W8A16 全量交付,投影 AP 73.90%,交付运行 NPU 0.516 ms/窗口 |
| [YOLOv8n](models/perception/object_detection/yolov8n/README.md) | Perception / Object detection | 640×640 RGB | Ultralytics v8.0.111 官方源码与权重 | Genio 720 完整交付,三后端全量结果已回收 |
| [Depth Anything V2 Small](models/navigation/single_camera_depth/depth_anything_v2_small/README.md) | Navigation / Single camera depth | 518×518 RGB | DepthAnything 官方源码与 Small 权重 | DA-2K 85.78%,ONNX 94.83%,精度变化 -9.04 个百分点,NPU 133.02 ms,峰值 RSS 33.04 MiB |
| [FastSAM-s](models/navigation/segmentation/fastsam/README.md) | Navigation / Segmentation | 640×640 RGB | CASIA-LMC-Lab / Ultralytics 8.0.111 | segm AR@100 37.57%,NPU 14.72 ms,峰值 RSS 64.55 MiB |
| [YOLOv5s](models/perception/object_detection/yolov5s/README.md) | Perception / Object detection | 640×640 RGB | Ultralytics | 本次全量结果已上传 |
| [ViT-Base Patch16 224](models/perception/image_classification/vit_base_patch16_224/README.md) | Perception / Image classification | 224×224 RGB | PyTorch Vision v0.15.1 | 本次全量结果已上传 |
| [RTMPose Body2d](models/interaction/pose_detection/rtmpose_body2d/README.md) | Interaction / Pose detection | 256×192 RGB | OpenMMLab MMPose v1.3.2 | WholeBody AP 53.24%,NPU 3.86 ms/人体框,峰值 RSS 32.74 MiB |
| [MobileFaceNet](models/interaction/face_recognition/mobilefacenet/README.md) | Interaction / Face recognition | 112×112 RGB 对齐人脸 | foamliu/MobileFaceNet | LFW 99.32%,NPU 0.52 ms,峰值 RSS 8.25 MiB |
| [Whisper-Tiny](models/audio/stt/whisper_tiny/README.md) | Audio / STT | 16 kHz 单声道、最长 30 秒 | OpenAI Whisper v20250625 | LibriSpeech test-clean 全量 WER 7.56%,NPU 379.19 ms/条音频,峰值 RSS 105.53 MiB |

公开示例结果见各模型 README; 当前指标依据已回收的板端 `results/summary.json` 更新.

各模型的当前数据集、精度和耗时见 README,机器可读结果统一保存为 `results/summary.json`。

模型文件大小统一保存在 `results/summary.json` 的 `model_size`,单位为 MB (十进制),
保留两位小数.记录 DLA 合计、外置 embedding、部署文件合计及文件明细,与运行时 RSS 区分.
已有模型可直接读取板端文件大小补录,无需重新转换、推理或评测; `source`、`directory`、
`collected_on`、`scope` 说明采集位置、日期和范围.板端没有的权重、ONNX、TFLite 为 `null`.
后续部署入口自动调用 `tools/model_size.py`,只补充大小字段,保留已有精度和性能记录.
多精度模型须明确选定本次部署文件,不合计历史副本; 新模型模板包含同样的要求.

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
| Docker 镜像 | Ubuntu 22.04 编译环境,镜像由本地配置指定 |
| Docker 容器 | 名称由平台配置指定 |
| Python | 当前容器 3.11.11 |
| NeuroPilot SDK | 8.0.11 |
| MTK Converter | 8.16.0 |
| Neuron Compiler | 8.2.31 |
| Genio 720 EVK | 连接地址由平台配置指定 |
| 板端目录 | 模型、数据集和结果位置由配置指定 |
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

首次使用前配置 `configs/storage.local.json`,填写模型和数据集根目录.已有本地配置时直接使用,不要用模板覆盖.

```bash
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
├── models/
│   ├── source_url.txt
│   ├── model_fp32.onnx
│   ├── model_fp16.onnx
│   ├── model_int8.tflite
│   └── model_int8.dla
├── deploy/
│   ├── board
│   ├── host
│   └── run.sh               # 编译主机与开发板共用的入口.
├── examples/
│   ├── input/
│   └── output/
└── results/
    └── summary.json
```

每个模型使用一个脚本完成两步流程：进入编译容器后运行 `deploy/run.sh` 编译并上传，再在开发板运行上传的同一脚本；路径在脚本中配置。具体命令见各模型 README。

模型产物与板端结果分别写入用户配置的 MODEL_OUTPUT_DIR 和 BOARD_RESULTS_DIR。

完整板端测试成功后保留 `summary.json` 和少量效果示例,包含平均 NPU 耗时、核心精度、匹配参考基准及精度差值。测试完成后将汇总上传到本地模型的 `results/summary.json`,效果文件取回 `examples/output/`,据此更新 README; 未上传结果不预填数字.

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

[MiniCPM5-2B](models/gen_ai/llm/minicpm5_2b/README.md) 轻量 Hessian 全 W4A16 已完成双 DLA、板端性能和首轮精度对比.固定 1000 题 MMLU 0-shot: FP32 55.9%,NPU 41.3%,下降 14.6 个百分点;WikiText 前 128 行 PPL: FP32 40.04,NPU 48.38.精度退化明显,不标记质量合格.

LLM 只交付 Prefill 与 Decode 两个 DLA,单位统一 token/s.本次 MiniCPM5 固定 128 输入、最多 128 输出的实测 Prefill 288.67 token/s,Decode 8.15 token/s;详细范围、哈希和排查证据见模型 README.

两款 LLM 保持全 W4A16.默认采用常规量化,不启用耗时累计层误差优化;已确认误差后只验证有限范围的局部修复.具体配置和实际质量状态见各模型 README.

### LLM 公共工具

两款 LLM 的公共校准、W4A16 量化、精度诊断和板端评测脚本统一位于 `tools/llm/`.模型目录保存各自配置、适配、部署入口与结果;调用方式和精度修复状态见各模型 README.参数量面向阅读使用 B/亿单位,精确值保留在机器可读配置中.

| 脚本 (`tools/llm/`) | 用途 | 执行位置 |
|---|---|---|
| `prepare_calibration.py` | 准备并记录固定校准样本 | 编译服务器 |
| `quantize_native.py` | 调用官方 W4A16 量化并核对实际位宽 | 编译服务器 |
| `evaluate_quality.py` | FP32/TFLite WikiText PPL,准备板端输入 | 编译服务器 |
| `compare_logits.py` | 对比 FP32/TFLite/NPU 输出误差 | 编译服务器 |
| `evaluate_mmlu.py` | 准备统一题目并计算 MMLU Accuracy | 服务器或开发板,按子命令选择 |
| `evaluate_board_quality.py` | MDLA WikiText PPL 与输出诊断 | 开发板 |
| `evaluate_native.py` | 原生运行工具的输出与 token/s 测试 | 开发板 |
## 模型与数据存储

仓库只保存代码、脚本、文档、公开来源记录和交付结果说明.原始权重、ONNX、量化模型、DLA 及配套运行参数统一保存在外部模型存储目录,数据集使用已有外部数据目录.

在项目根目录复制 `configs/storage.example.json` 为 `configs/storage.local.json`,填写 `models_root` 和 `dataset_root` 的实际绝对路径.本地配置被 Git 忽略,完整内部路径不能写入提交的脚本、文档或报告.也可设置 `MTK_MODEL_STORAGE_ROOT` 和 `MTK_DATASET_ROOT`;环境变量优先于本地配置.

映射规则: 代码目录 `models/场景/类别/模型名/` 对应外部 `models_root/场景/类别/模型名/`.原 `模型名/models/` 的资源直接放入该外部目录,不额外增加一层 `models/`.模型源码和公开来源记录仍保留在仓库.转换和编译产物写入外部资源目录,临时缓存保存在各模型配置的构建目录.

```bash
bash docker/create_container.sh
bash docker/enter_container.sh
```

容器读写挂载模型目录、只读挂载数据集目录,挂载前后使用同一个绝对路径.已有容器缺少挂载时先保存现有环境,再重建;本地配置可通过 `container_image` 指定保存的环境镜像.不自动下载、安装或运行模型测试.

各模型原有 `deploy/run.sh` 和 LLM `deploy/run_native.sh` 从本地配置读取资源路径.FastSAM 的 `config.yaml` 保留模型参数和相对数据路径,实际存储根目录也由本地配置提供.板端继续使用上传的独立部署配置和板端模型目录.

FastSAM 编译步骤按仓库真实目录导入 `tools.storage` 和 `models.navigation.segmentation.fastsam.deploy.board.fastsam_utils`.进入容器后在仓库根目录,使用容器的 Python 执行一次,注册仓库根目录:

```bash
python3 -m pip install --no-index --no-deps --no-build-isolation --config-settings editable_mode=compat -e .
```

使用环境已有的 setuptools>=64,不下载模型依赖.可编辑安装直接使用当前源码,移动仓库后需要重新安装.VS Code 选择完成安装的同一个 Python 解释器.

LLM 资源目录中,`original/` 保存官方权重和 tokenizer,`native/` 保存当前原生转换及量化资源,`compiled/版本名/` 保存编译包的归档副本.已完成交付的模型包路径沿用该模型 README 的记录.历史精度和交付状态保持原记录,文件迁移不代表重新验证模型.

一次性迁移工具为 `tools/migrate_model_storage.py`.先记录迁移清单,确认后使用 `--apply`;每个文件在移除来源前核对大小与 SHA256,同名不同内容的旧目标保存在 `_previous/哈希/` 下.来源不同且内容不一致的模型应分别保留并记录来源.迁移记录应保存在仓库外,不能提交包含内部路径的清单.

G720 的板端连接与工具链配置统一位于 `configs/platforms/genio_720.yaml`,SDK 版本、路径、交叉编译器和容器名称在其 `toolchain` 分组维护.执行 `docker/enter_container.sh` 后,容器终端自动加载平台和资源环境,无需手动执行初始化脚本; 修改环境配置后重新进入终端即可,无需重启容器.FastSAM 的 C++ 构建改由 Makefile 使用这些环境变量完成,不再使用 Python 包装编译命令.


## 公共代码

模型只保留自身的输入约定、解码和评测策略,重复的基础功能放在 `tools/`:

| 位置 | 职责 |
| --- | --- |
| `tools/files.py`、`tools/process.py` | 文件校验、分块哈希、JSON 读取和命令执行 |
| `tools/boxes.py`、`tools/boxes.h` | 矩形框 IoU 等基础计算 |
| `tools/quantization.py` | 逐张量整数 I/O 元数据校验 |
| `tools/coco.py` | COCO 检测评测,模型入口指定名称和结果状态规则 |
| `tools/board/` | C++ 文件和统计工具、检测结果处理、共享性能测试源码 |
| `tools/llm/package.py`、`tools/llm/run_native.sh` | 原生 LLM 张量契约与部署入口 |

部署入口会将板端需要的公共 Python 模块一起复制到部署目录,无需板端访问代码仓库.模型相关的解码、预处理约定和平台适配仍由模型自身维护.

公共 Python 回归测试可在编译容器内执行 `python3 -m unittest discover -s tests -p 'test_*.py'`.C++ 回归测试源码为 `tests/common_cpp_test.cpp`,交叉编译后在目标开发板执行;两者均不需要模型权重或全量数据集.LLM 契约测试使用编译容器已有的 Converter 依赖.
