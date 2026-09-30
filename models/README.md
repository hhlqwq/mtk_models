# 模型目录与部署路线图

本目录按照“Scenario → Category → Model”三级结构组织模型，覆盖模型来源、转换、量化、
MTK NPU 部署、板端 Demo、精度与性能评估以及文档交付。Qualcomm AI Hub Models 仅作为
目录结构、模型卡、Demo 和评测方式的 `delivery_reference`，实际模型必须使用官方或原作者
公开源码与权重完成 MTK 侧转换和验证。

## 部署总览

> 数据快照：2026-09-29。已注册模型统计来自 `registry/models.yaml` 和
> `registry/target_models.yaml`。

| 指标 | 数量 | 说明 |
| --- | ---: | --- |
| 已进入实现注册表 | 8 / 46（17.4%） | 8 个注册实现相对于 46 个目标候选的规模比；存在模型变体映射和额外 Model Zoo 模型，不等同于严格完成率 |
| Genio 720 完整交付 | 3 / 8（37.5%） | 精度、性能、Demo 和文档均已闭环 |
| Genio 720 已注册模型的板端运行证据 | 8 / 8（100%） | FastSAM-s 三端同协议 COCO 全量精度已核对；注册状态包括 `complete` 与 `board_verified` |
| Genio 5100 已开始 | 0 / 8（0.0%） | 当前所有注册模型均为 `not_started` |

### Genio 720 状态分布

| 状态 | 数量 | 占已注册模型比例 |
| --- | ---: | ---: |
| 完整交付 `complete` | 3 | 37.5% |
| 板端验证 `board_verified` | 5 | 62.5% |

## 已注册模型状态

| 模型 | 机器人能力 | 开源实现来源 | Genio 720 | Genio 5100 | Inference Time | 评测数据集 | Reference Metric | On-device Metric | 当前缺口与下一步 |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| [Depth Anything V2 Small](navigation/single_camera_depth/depth_anything_v2_small/README.md) | 单目相对深度与空间结构 | DepthAnything/Depth-Anything-V2 | 🔵 `board_verified` | ⚪ `not_started` | [常驻 Runtime 调用均值 133.902 ms/图](navigation/single_camera_depth/depth_anything_v2_small/docs/benchmark.md) | DA-2K，1,033 图、2,068 点对 | [FP32 ONNX DA-2K 点对准确率 94.83%](navigation/single_camera_depth/depth_anything_v2_small/docs/accuracy.md) | [NPU DA-2K 点对准确率 85.78%](navigation/single_camera_depth/depth_anything_v2_small/docs/accuracy.md) | 同协议全量对照与常驻性能已测；板端低 9.04 个百分点，需分析量化误差 |
| [FastSAM-s](navigation/segmentation/fastsam/README.md) | 目标分割、交互式区域选择 | CASIA-LMC-Lab/FastSAM | 🔵 `board_verified` | ⚪ `not_started` | [独立校准 DLA 常驻 Runtime 调用均值 14.0536 ms/图](navigation/segmentation/fastsam/docs/benchmark.md) | COCO val2017，5,000 图；类别无关分割 | [PyTorch segm AR@100 0.391；ONNX 0.390](navigation/segmentation/fastsam/docs/accuracy.md) | [独立校准 DLA AR@100 0.376384](navigation/segmentation/fastsam/docs/accuracy.md) | 独立校准结果较参考端低约 0.014 |
| [YOLOv5s](perception/object_detection/yolov5s/README.md) | 通用目标检测 | Ultralytics/YOLOv5 | 🟢 `complete` | ⚪ `not_started` | [9.629 ms/图](perception/object_detection/yolov5s/README.md#历史性能) | COCO val2017，5,000 图；bbox | [FP32 ONNX mAP@0.5:0.95 0.3709](perception/object_detection/yolov5s/README.md#历史精度) | [INT8 mAP@0.5:0.95 0.3586](perception/object_detection/yolov5s/README.md#历史精度) | 作为 Genio 5100 首个迁移基线，复用已完成的全链路验收协议 |
| [ViT-Base Patch16 224](perception/image_classification/vit_base_patch16_224/README.md) | 图像分类 | PyTorch Vision | 🟢 `complete` | ⚪ `not_started` | [51.7129 ms/图](perception/image_classification/vit_base_patch16_224/README.md#历史性能) | ILSVRC2012 val，50,000 图 | [FP32 ONNX Top-1 80.64%](perception/image_classification/vit_base_patch16_224/README.md#历史精度) | [INT8 Top-1 79.38%](perception/image_classification/vit_base_patch16_224/README.md#历史精度) | Genio 720 已闭环；后续按平台需求迁移 Genio 5100 |
| [RTMPose Body2d](interaction/pose_detection/rtmpose_body2d/README.md) | 人体姿态和人机交互 | OpenMMLab/MMPose | 🟢 `complete` | ⚪ `not_started` | [3.8527 ms/人体框](interaction/pose_detection/rtmpose_body2d/docs/benchmark.md) | COCO-WholeBody V1.0 val，5,000 图、104,125 人体框 | [FP32 ONNX WholeBody AP 0.5703](interaction/pose_detection/rtmpose_body2d/docs/accuracy.md) | [INT8 WholeBody AP 0.5324](interaction/pose_detection/rtmpose_body2d/docs/accuracy.md) | 单框耗时不含上游人体检测器；后续可组合动作识别 |
| [MobileFaceNet](interaction/face_recognition/mobilefacenet/README.md) | 人脸特征提取与身份匹配 | foamliu/MobileFaceNet | 🔵 `board_verified` | ⚪ `not_started` | [旧 DLA 常驻 Runtime 调用均值 0.476406 ms/图](interaction/face_recognition/mobilefacenet/docs/benchmark.md) | LFW，6,000 对、10 折；对齐协议重测中 | [FP32 ONNX 待重测](interaction/face_recognition/mobilefacenet/docs/accuracy.md) | [NPU 待重测](interaction/face_recognition/mobilefacenet/docs/accuracy.md) | 旧结果输入协议错误；修正后全量精度与性能待完成 |
| [YOLO-World XL](perception/object_detection/yoloworld_xl/README.md) | 开放词汇目标检测 | AILab-CVC/YOLO-World | 🔵 `board_verified` | ⚪ `not_started` | [混合 EP `session.Run` 均值 3864.08 ms/图](perception/object_detection/yoloworld_xl/docs/benchmark.md) | COCO val2017，5,000 图；bbox | [FP32 ONNX COCO bbox AP50:95 0.472953](perception/object_detection/yoloworld_xl/docs/accuracy.md) | [混合 EP COCO bbox AP50:95 0.472952](perception/object_detection/yoloworld_xl/docs/accuracy.md) | 历史全量结果为混合 EP；纯 NPU 正确性未通过，暂停后续全量测试 |
| [Whisper-Tiny](audio/stt/whisper_tiny/README.md) | 中英文语音识别 | OpenAI/Whisper | 🔵 `board_verified` | ⚪ `not_started` | [NPU 调用均值 460.05 ms/段音频](audio/stt/whisper_tiny/docs/benchmark.md) | LibriSpeech `test-clean`，2,620 条 | [OpenAI CUDA `test-clean` WER 7.5546%](audio/stt/whisper_tiny/docs/accuracy.md) | [NPU `test-clean` WER 7.5603%](audio/stt/whisper_tiny/docs/accuracy.md) | `test-clean` 2,620 条全量对照完成；AISHELL-1 CER 为历史结果，其他交付项按模型文档核查 |

`Inference Time` 列列出各报告实际测得的计时口径，并非统一的纯 NPU 延迟。YOLOv5s 和 RTMPose
取板端常驻 C++ 流程中的 NPU 阶段；ViT 为连续 100 次纯 NPU 推理；FastSAM 与 Whisper-Tiny
为 Runtime 调用耗时；YOLO-World XL 为 Neuron/CPU 混合 EP 的 `session.Run`；Depth Anything
与 MobileFaceNet 为预热后 100 次常驻 C++ `NeuronRuntime_inference` API 调用；这两者另有逐图 CLI 墙钟记录。不同协议的耗时不能直接排名。

`Reference Metric` 只填写与对应板端结果使用相同数据集和协议的浮点模型基线。
FastSAM 旧参考端与板端置信度阈值不同，现已使用统一协议完成三端全量复评；其余模型的
机器可读结果位于各模型 `results/reference_accuracy/`。Whisper 使用 OpenAI CUDA
FP16 计算，其余已完成参考端使用 FP32 ONNX CUDA 优先执行。
`On-device Metric` 为真实板端全量指标。YOLOv5s 使用 COCO val2017 5,000 图，ViT 使用
ImageNet val 50,000 图，RTMPose 使用 COCO-WholeBody；Whisper-Tiny 本轮正式 WER 仅使用
LibriSpeech `test-clean` 2,620 条，旧 AISHELL-1 CER 另见其模型精度文档。FastSAM 的类别无关
分割、Depth Anything 的固定方形缩放及 MobileFaceNet 的非对齐 LFW 均有各自协议限制，见对应模型文档。
本轮参考端的 COCO 5,000 张、DA-2K 1,033 张、LFW 用到的 7,701 张及 LibriSpeech
2,620 条音频，四组结果均已记录在对应板端报告中。

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

每个模型目录至少应包含下列内容；YOLOv5s 将来源记录和离线包放在 `models/`，无需单独的 `original/`：

```text
<model>/
├── README.md
├── model_card.md
├── LICENSE
├── original/             # 或在 models/ 内保存来源记录与离线包.
├── models/
├── deploy/
├── examples/
└── docs/
```

完整交付必须形成以下证据链：

1. 使用官方或原作者源码与权重，并记录版本、下载地址、许可证和文件大小。
2. 保存原始框架推理结果，自行导出 ONNX 或 TFLite，并完成数值一致性比较。
3. 完成 MTK 量化和 NPU 编译，记录工具链、命令、输入输出及量化配置。
4. 在目标开发板执行真实模型推理和可复现 Demo，而不是只检查 Runtime 或模型文件。
5. 使用正式数据集比较原始框架、ONNX/TFLite 和 MTK NPU 的同协议精度。
6. 报告模型耗时、端到端耗时、P50/P90/P95、FPS、峰值 RSS 和 NPU/CPU 执行边界。
7. 更新模型 README、模型卡、精度报告、性能报告、示例输出和注册表状态。

对于机器人应用，还应在模型指标之外记录应用层输出，例如可通行区域、目标相对深度、
稳定 `track_id`、手势防抖结果或环境声告警；单张可视化图片不能替代连续运行和安全边界验证。
