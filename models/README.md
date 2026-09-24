# 模型目录与部署路线图

本目录按照“Scenario → Category → Model”三级结构组织模型，覆盖模型来源、转换、量化、
MTK NPU 部署、板端 Demo、精度与性能评估以及文档交付。Qualcomm AI Hub Models 仅作为
目录结构、模型卡、Demo 和评测方式的 `delivery_reference`，实际模型必须使用官方或原作者
公开源码与权重完成 MTK 侧转换和验证。

## 部署总览

> 数据快照：2026-09-24。已注册模型统计来自 `registry/models.yaml` 和
> `registry/target_models.yaml`；未注册实现另按模型目录中的 `model.yaml` 与板端报告列出。

| 指标 | 数量 | 说明 |
| --- | ---: | --- |
| 已进入实现注册表 | 6 / 46（13.0%） | 6 个注册实现相对于 46 个目标候选的规模比；存在模型变体映射和额外 Model Zoo 模型，不等同于严格完成率 |
| 已有模型目录但未注册 | 1 | MobileFaceNet 已板端冒烟，尚未加入 `registry/models.yaml` |
| Genio 720 完整交付 | 3 / 6（50.0%） | 精度、性能、Demo 和文档均已闭环 |
| Genio 720 已注册模型的板端运行证据 | 6 / 6（100%） | FastSAM 已完成单图硬件冒烟；还包括 `complete`、`board_verified` 和 Whisper 的历史非规范状态 `board_validated` |
| Genio 720 未注册模型的板端运行证据 | 1 / 1 | MobileFaceNet 已完成三次硬件冒烟，正式指标待测 |
| Genio 5100 已开始 | 0 / 6（0.0%） | 当前所有注册模型均为 `not_started` |

### Genio 720 状态分布

| 状态 | 数量 | 占已注册模型比例 |
| --- | ---: | ---: |
| 完整交付 `complete` | 3 | 50.0% |
| 板端验证 `board_verified` | 2 | 33.3% |
| 历史状态 `board_validated` | 1 | 16.7% |

## 已注册模型状态

| 模型 | 机器人能力 | 官方来源 | Genio 720 | Genio 5100 | Inference Time | Reference Metric | On-device Metric | 当前缺口与下一步 |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| [FastSAM-s](navigation/segmentation/fastsam/README.md) | 目标分割、交互式区域选择 | CASIA-LMC-Lab/FastSAM | 🔵 `board_verified` | ⚪ `not_started` | [单次 15.909 ms/图](navigation/segmentation/fastsam/docs/board_smoke_20260923.md) | 待正式评测 | 待正式评测 | 官方权重已完成转换、编译与单图硬件冒烟；补齐正式类别无关分割精度与预热后稳定性能 |
| [YOLOv5s](perception/object_detection/yolov5s/README.md) | 通用目标检测 | Ultralytics/YOLOv5 | 🟢 `complete` | ⚪ `not_started` | [9.629 ms/图](perception/object_detection/yolov5s/docs/benchmark.md) | [FP32 ONNX mAP@0.5:0.95 0.3709](perception/object_detection/yolov5s/docs/accuracy.md) | [INT8 mAP@0.5:0.95 0.3586](perception/object_detection/yolov5s/docs/accuracy.md) | 作为 Genio 5100 首个迁移基线，复用已完成的全链路验收协议 |
| [ViT-Base Patch16 224](perception/image_classification/vit_base_patch16_224/README.md) | 图像分类 | PyTorch Vision | 🟢 `complete` | ⚪ `not_started` | [51.7129 ms/图](perception/image_classification/vit_base_patch16_224/docs/benchmark.md) | [FP32 ONNX Top-1 80.64%](perception/image_classification/vit_base_patch16_224/docs/accuracy.md) | [INT8 Top-1 79.38%](perception/image_classification/vit_base_patch16_224/docs/accuracy.md) | Genio 720 已闭环；后续按平台需求迁移 Genio 5100 |
| [RTMPose Body2d](interaction/pose_detection/rtmpose_body2d/README.md) | 人体姿态和人机交互 | OpenMMLab/MMPose | 🟢 `complete` | ⚪ `not_started` | [3.8527 ms/人体框](interaction/pose_detection/rtmpose_body2d/docs/benchmark.md) | [FP32 ONNX WholeBody AP 0.5703](interaction/pose_detection/rtmpose_body2d/docs/accuracy.md) | [INT8 WholeBody AP 0.5324](interaction/pose_detection/rtmpose_body2d/docs/accuracy.md) | 单框耗时不含上游人体检测器；后续可组合动作识别 |
| [YOLO-World XL](perception/object_detection/yoloworld_xl/README.md) | 开放词汇目标检测 | AILab-CVC/YOLO-World | 🔵 `board_verified` | ⚪ `not_started` | [3720.758 ms/图](perception/object_detection/yoloworld_xl/docs/benchmark.md)，混合 EP | 待正式评测 | 待正式评测 | 当前耗时来自 3 图、9 次 `session.run`；补齐正式 COCO 精度与性能评测 |
| [Whisper-Tiny](audio/stt/whisper_tiny/README.md) | 中英文语音识别 | OpenAI/Whisper | 🟣 `board_validated`¹ | ⚪ `not_started` | [296.899 ms/段音频](audio/stt/whisper_tiny/docs/benchmark.md) | [OpenAI CUDA CER 45.8264%](audio/stt/whisper_tiny/docs/accuracy.md) | [NPU CER 45.5935%](audio/stt/whisper_tiny/docs/accuracy.md) | AISHELL-1 test 7,176 条；绝对 CER 较高，LibriSpeech WER 待测，并需统一注册表状态 |

### 未进入注册表的实现

| 模型 | 机器人能力 | Genio 720 | Genio 5100 | Inference Time | Reference Metric | On-device Metric | 注册与评测缺口 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [MobileFaceNet](interaction/face_recognition/mobilefacenet/README.md) | 人脸特征提取与身份匹配 | 🔵 `board_verified`（`model.yaml`） | ⚪ `not_started` | [待正式评测](interaction/face_recognition/mobilefacenet/docs/benchmark.md) | [待正式评测](interaction/face_recognition/mobilefacenet/docs/accuracy.md) | [待正式评测](interaction/face_recognition/mobilefacenet/docs/accuracy.md) | [三次硬件冒烟已通过](interaction/face_recognition/mobilefacenet/docs/smoke.md)；尚未加入 `registry/models.yaml`，未完成身份验证精度和稳定耗时评测 |

`Inference Time` 为 Genio 720 板端模型推理平均耗时，不包含前后处理。YOLOv5s 和 RTMPose
取板端常驻 C++ 流程中的 NPU 阶段；ViT 为连续 100 次纯 NPU 推理；YOLO-World XL 为
Neuron EP 与 CPU EP 混合执行的 `session.run` 耗时，不能视为纯 NPU 延迟；Whisper-Tiny
为 Encoder 与自回归 Decoder 的 NPU 调用总和，随音频和生成 Token 数变化。不同任务的
每图、每人体框、每段音频耗时不能直接比较。FastSAM 的数值仅为一次 C++ 硬件 API
调用耗时，未经预热和重复统计，不能作为稳定平均延迟。MobileFaceNet 尚无毫秒级计时报告。

`Reference Metric` 是本项目同协议浮点模型精度参考，`On-device Metric` 是实际板端模型
推理得到的精度；两列使用相同任务指标与评测集。YOLOv5s 使用 COCO val2017 5,000 图，
ViT 使用 ImageNet val 50,000 图，RTMPose 使用 COCO-WholeBody 验证集与相同人体框，
Whisper-Tiny 使用 AISHELL-1 test 7,176 段音频。Whisper 的 CER 越低越好，但当前数值
不代表精度验收通过。尚未实施的推荐模型，以及 FastSAM、YOLO-World XL 和
MobileFaceNet 的正式精度，均不填写估算值。

¹ `board_validated` 是 `registry/models.yaml` 当前保存的历史状态值，但不属于
`registry/compatibility.yaml` 和 `registry/schema.json` 定义的合法集合。在修正注册表前，
本文保留原值，避免把非标准状态误写成 `complete`。

### 标准状态说明

| 状态 | 含义 |
| --- | --- |
| ⚪ `not_started` | 尚未开始 |
| 🟡 `environment_ready` | 环境与目录已准备，尚未完成 MTK 部署模型转换 |
| 🟠 `converted` | 已生成 MTK 部署模型，尚未完成目标板验证 |
| 🔵 `board_verified` | 已在目标板完成真实模型推理，精度、性能或文档仍未闭环 |
| 🟢 `complete` | 精度、性能、Demo 和文档已完整交付 |
| 🔴 `unsupported` | 已确认当前工具链或硬件不支持，并保留失败证据 |

静态图检查、转换成功、`neuronrt -v` 或编译命令成功都不能替代真实板端推理；只有保存
输入输出、运行日志、性能、精度和资源占用证据后，才能更新对应状态。

## 机器人模型推荐路线图

推荐顺序以“补齐机器人能力闭环、复用现有工程资产、控制 MTK 转换风险”为原则，不按模型
热度排序。

| 优先级 | 推荐模型或工作项 | 目标能力 | 建议目录 | 推荐理由 | 进入实现前的关键门槛 |
| --- | --- | --- | --- | --- | --- |
| P0 | DeepLabV3-Plus-MobileNet | 地面、道路、墙体和障碍物语义分割 | `navigation/segmentation/deeplabv3_plus_mobilenet` | 轻量 CNN、固定输入输出，适合先建立机器人可通行区域闭环 | 锁定官方源码、权重、许可证、数据集和类别映射；先检查空洞卷积与 resize 算子 |
| P0 | YOLOv5s → Genio 5100 | 第二平台部署基线 | 复用现有 YOLOv5s 包 | Genio 720 证据最完整，最适合验证框架的平台抽象与兼容性 | 核对 G5100 SDK、编译架构、Runtime、量化和板端部署差异 |
| P1 | Depth Anything V2 Small | 单目相对深度、障碍物远近和空间结构 | `navigation/single_camera_depth/depth_anything_v2_small` | 能把二维检测扩展为空间感知，直接服务移动机器人导航 | 固定输入尺寸；检查 Transformer、插值和量化误差；不得把相对深度直接当作安全米制距离 |
| P1 | MediaPipe Hand Gesture | 手检测、21 点关键点和手势控制 | `interaction/gesture/mediapipe_hand_gesture` | 能形成直观的人机交互 Demo，并补足现有 RTMPose 的手部能力 | 按子模型分别验证检测、关键点和分类；动作控制必须使用多帧确认与超时失效 |
| P1 | YAMNet | 警报、撞击、玻璃破碎和机械异常声识别 | `audio/ambient_sound/yamnet` | MobileNetV1 架构轻量，可与 Whisper 形成“语音内容 + 环境事件”双通路 | 明确音频窗口、步长、类别子集和流式缓存；使用真实机器人环境噪声验证 |
| P2 | OSNet + BoxMOT | 人员持续跟踪和身份保持 | `slam/object_tracking/boxmot_osnet` | 可复用现有 YOLO 检测结果，ReID 上 NPU、关联和 Kalman Filter 留在 CPU | 分开统计 Detector、ReID 和 Tracker 的耗时与资源，不能把 BoxMOT 当作单一 NPU 模型 |
| P2 | SuperPoint → LightGlue | 视觉定位、特征匹配和 SLAM | `slam/localization/superpoint_lightglue` | 补齐机器人定位能力，适合按两个独立模型分阶段推进 | 先固定最大关键点数验证 SuperPoint；LightGlue 的动态匹配和注意力先允许 CPU 执行 |
| P3 | FastSAM / EdgeSAM / MobileSAM | 点选目标、抓取区域和开放分割 | 复用 FastSAM，其他模型暂留目标清单 | 适合机械臂和交互展示，但不应替代持续运行的基础语义分割 | 评估提示输入、多阶段图、内存占用和许可证限制 |
| P3 | PointPillar-Tiny / CenterPoint | 3D 点云障碍物检测 | `pointcloud/pointcloud/<model>` | 面向带激光雷达的机器人，长期价值高 | 先确定雷达硬件、点云格式、标定、数据集、体素化和 3D 后处理边界 |

### 建议实施顺序

```text
阶段 1：DeepLabV3-Plus-MobileNet / Genio 720
   └── 同步建立 YOLOv5s / Genio 5100 平台基线

阶段 2：Depth Anything V2 Small
   ├── 检测框 + 相对深度融合
   └── 可通行区域 + 深度风险融合

阶段 3：MediaPipe Hand Gesture 或 YAMNet
   └── 根据产品更偏人机交互还是环境安全感知二选一

阶段 4：OSNet + BoxMOT
   └── 复用 YOLO，形成稳定人员跟踪和跟随能力

阶段 5：SuperPoint，再评估 LightGlue
   └── 进入视觉定位和 SLAM 能力建设
```

## 目录结构

```text
models/
├── perception/
│   ├── object_detection/
│   └── image_classification/
├── interaction/
│   ├── gesture/
│   ├── hand_landmark/
│   ├── face_recognition/
│   ├── pose_detection/
│   └── text_recognition/
├── navigation/
│   ├── segmentation/
│   ├── single_camera_depth/
│   ├── dual_camera_depth/
│   └── vision_foundation_model/
├── gen_ai/
│   ├── vision_language/
│   ├── llm/
│   └── vlm/
├── audio/
│   ├── stt/
│   ├── tts/
│   └── ambient_sound/
├── slam/
│   ├── object_tracking/
│   └── localization/
├── pointcloud/
│   └── pointcloud/
└── three_d/
    └── three_d_fusion/
```

未开始的模型只保留在 `registry/target_models.yaml`，不预先创建大量空目录。模型通过
来源、许可证、图结构和 MTK 算子预检查并进入实现阶段后，使用 `tools/create_model.py`
创建完整交付目录，再加入 `registry/models.yaml`。

## 单模型交付要求

每个模型目录至少应包含：

```text
<model>/
├── README.md
├── model_card.md
├── model.yaml
├── LICENSE
├── original/
├── models/
├── deploy/
├── examples/
└── docs/
```

完整交付必须形成以下证据链：

1. 使用官方或原作者源码与权重，并记录版本、下载地址、许可证、文件大小和 SHA-256。
2. 保存原始框架推理结果，自行导出 ONNX 或 TFLite，并完成数值一致性比较。
3. 完成 MTK 量化和 NPU 编译，记录工具链、命令、输入输出、量化配置和产物哈希。
4. 在目标开发板执行真实模型推理和可复现 Demo，而不是只检查 Runtime 或模型文件。
5. 使用正式数据集比较原始框架、ONNX/TFLite 和 MTK NPU 的同协议精度。
6. 报告模型耗时、端到端耗时、P50/P90/P95、FPS、峰值 RSS 和 NPU/CPU 执行边界。
7. 更新模型 README、模型卡、精度报告、性能报告、示例输出和注册表状态。

对于机器人应用，还应在模型指标之外记录应用层输出，例如可通行区域、目标相对深度、
稳定 `track_id`、手势防抖结果或环境声告警；单张可视化图片不能替代连续运行和安全边界验证。
