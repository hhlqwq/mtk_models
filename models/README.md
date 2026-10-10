# 模型目录与部署路线图

本目录按照“Scenario → Category → Model”三级结构组织模型，覆盖模型来源、转换、量化、
MTK NPU 部署、板端 Demo、精度与性能评估以及文档交付。Qualcomm AI Hub Models 仅作为
目录结构、模型卡、Demo 和评测方式的 `delivery_reference`，实际模型必须使用官方或原作者
公开源码与权重完成 MTK 侧转换和验证。

## 模型规格

本次补充 LLM: [DeepSeek-R1-Distill-Qwen-1.5B](gen_ai/llm/deepseek_r1_distill_qwen_1_5b/README.md).
目标为中英文文本问答与推理,来源为 DeepSeek 官方权重,地瓜 RDK 仅作为部署参考.
Genio 720 已完成 FP16 双语 Demo 与三后端样例数值检查 (`board_verified`),
Genio 5100 尚未开始.W4A16 与正式基准精度尚未验证.

输入为导出模型的张量形状,图像统一按 `N×C×H×W` 表示,批量大小为 1。音频模型的采样率均为 16 kHz。

| 模型 | 具体版本 / 权重 | 模型输入 | 参数量 (M) | FP32 ONNX 大小 (MB) | 部署 DLA 大小 (MB) |
| --- | --- | --- | ---: | ---: | --- |
| [DeepSeek-R1-Distill-Qwen-1.5B](gen_ai/llm/deepseek_r1_distill_qwen_1_5b/README.md) | DeepSeek 官方固定 revision,实际参数包含独立 LM Head | Token + KV Cache,上下文 1024,28 层 | 1777.09 | 6175.24 (38 个分片合计) | FP16 **3089.46**,另需 embedding **466.75** |
| [YAMNet](audio/ambient_sound/yamnet/README.md) | Google AudioSet 521 类,源码 `34a2132` | Log-Mel `1×1×96×64`,单声道音频 | 3.73 | 14.94 | W8A16 **3.99** |
| [YOLOv8n](perception/object_detection/yolov8n/README.md) | YOLOv8-N,Ultralytics `v8.0.111` | RGB `1×3×640×640` | 3.16 | 12.66 | INT8 **3.53** |
| [YOLOv5s](perception/object_detection/yolov5s/README.md) | YOLOv5-S,官方 `v7.0` 权重 | RGB `1×3×640×640` | 7.24 | 28.94 | INT8 **7.66** |
| [ViT-Base Patch16 224](perception/image_classification/vit_base_patch16_224/README.md) | ViT-B/16,TorchVision `IMAGENET1K_V1` | RGB `1×3×224×224` | 86.57 | 346.44 | INT8 **88.00** |
| [RTMPose Body2d](interaction/pose_detection/rtmpose_body2d/README.md) | RTMPose-M WholeBody 133 点,MMPose `v1.3.2` | RGB 人体裁剪 `1×3×256×192` | 17.95 | 71.83 | INT8 **19.14** |
| [MobileFaceNet](interaction/face_recognition/mobilefacenet/README.md) | foamliu MobileFaceNet,`v1.0` 权重,128 维特征 | RGB 对齐人脸 `1×3×112×112` | 1.00 | 4.00 | INT8 **1.41** |
| [Depth Anything V2 Small](navigation/single_camera_depth/depth_anything_v2_small/README.md) | Depth Anything V2 Small,ViT-S/14 | RGB `1×3×518×518` | 24.79 | 98.97 | INT8 **25.83** |
| [FastSAM-s](navigation/segmentation/fastsam/README.md) | 官方 `FastSAM-s.pt`,Ultralytics `v8.0.111` | RGB `1×3×640×640` | 11.79 | 47.18 | INT8 **12.29** |
| [Whisper-Tiny](audio/stt/whisper_tiny/README.md) | OpenAI 多语言 `tiny`,源码 `v20250625` | encoder: Log-Mel `1×80×3000`; decoder: token + KV cache | 37.18 | 230.79 (双模型合计) | 浮点 **116.03** (双模型合计) |

参数量按当前原始网络统计,`1 M = 100 万个参数`,不计 BatchNorm 运行统计量等缓冲区。文件大小来自当前生成文件,`1 MB = 1,000,000 bytes`,与运行时峰值内存分开记录。

Whisper 参数量统计原始完整网络,encoder / decoder 文件大小分别为 ONNX **32.87 / 197.92 MB**、DLA **16.60 / 99.43 MB**。模型来源与完整版本见各模型的 `model_card.md` 和 `models/source_url.txt`。

## 当前测试结果

根据已回收的各模型板端 `results/summary.json` 更新。参考基准必须与板端模型及评测协议匹配; 未上传结果的模型不预填数字。

| 模型 | 板端 NPU 平均耗时 (ms) | 峰值 RSS (MiB) | 核心精度 | 精度变化 (百分点) |
| --- | ---: | ---: | --- | ---: |
| [DeepSeek FP16](gen_ai/llm/deepseek_r1_distill_qwen_1_5b/README.md) | **220.11/Token** | **148.10** | 自编双语 26 Token 的 PPL **145.05**,非正式精度 | 不适用 |
| [YAMNet W8A16](audio/ambient_sound/yamnet/README.md) | **0.516** | **10.75** | ESC-50 47 类投影宏平均 AP **73.8964%** | **-0.2491** |
| [YOLOv8n](perception/object_detection/yolov8n/README.md) | **6.28** | **31.32** | mAP@0.5:0.95 **35.35%** | **-1.275** |
| [YOLOv5s](perception/object_detection/yolov5s/README.md) | **9.76** | **33.23** | mAP@0.5:0.95 **35.86%** | **-1.23** |
| [ViT-Base Patch16 224](perception/image_classification/vit_base_patch16_224/README.md) | **51.75** | **141.30** | Top-1 **79.40%** | **-1.27** |
| [RTMPose Body2d](interaction/pose_detection/rtmpose_body2d/README.md) | **3.86** | **32.74** | WholeBody AP **53.24%** | **-3.80** |
| [MobileFaceNet](interaction/face_recognition/mobilefacenet/README.md) | **0.52** | **8.25** | LFW 验证准确率 **99.32%** | **-0.07** |
| [Depth Anything V2 Small](navigation/single_camera_depth/depth_anything_v2_small/README.md) | **133.02** | **33.04** | DA-2K 点对准确率 **85.78%** | **-9.04** |
| [FastSAM-s](navigation/segmentation/fastsam/README.md) | **14.72** | **64.55** | segm AR@100 **37.57%** | **-1.45** |
| [Whisper-Tiny](audio/stt/whisper_tiny/README.md) | **379.19/条音频** | **105.53** | WER **7.56%** | 下降不足 0.01 |

DeepSeek 结果来自运行 `20261010_deepseek_fp16_v3`,batch=1、上下文 1024、FP16、
逐 Token Prefill、Greedy 生成上限 512.中文 / 英文 Demo 分别在 73 / 412 Token 到达 EOS,
TTFT 为 3.245 / 3.112 秒,端到端 Decode 为 3.78 / 3.81 Token/s.
表中 220.11 ms 为模型算子的纯 NPU 调用平均耗时; RSS 未覆盖全部 NPU 驱动内存,
本次系统可用内存下降约 3512.09 MiB.三后端自编短文本 PPL 为 144.57 / 144.57 / 145.05,
仅验证数值,不能替代正式基准.结果与实际文本见 [DeepSeek README](gen_ai/llm/deepseek_r1_distill_qwen_1_5b/README.md).

YOLOv5s 结果来自运行 `20261008_032807_67603`,COCO val2017 全量 5000 张; ONNX FP32 与板端精度均为本次同协议实测,ONNX mAP 为 **37.09%**。结果文件见 [summary.json](perception/object_detection/yolov5s/results/summary.json)。

ViT 结果来自运行 `20261008_094211_126146`,ImageNet val2012 全量 50000 张; 同协议 ONNX FP32 Top-1 为 **80.66%**,板端为 **79.40%**。结果文件见 [summary.json](perception/image_classification/vit_base_patch16_224/results/summary.json)。

YOLOv8n 结果来自运行 `20261008_yolov8n_full_v1`,COCO val2017 全量 5000 张;
PyTorch / ONNX / NPU mAP 分别为 **36.6444% / 36.6260% / 35.3510%**,
板端端到端平均耗时为 **22.925 ms**,P95 为 **27.088 ms**.
量化损失使用本次同协议 ONNX 作为基准,详见
[YOLOv8n README](perception/object_detection/yolov8n/README.md).

YAMNet 交付方案为全 W8A16,运行 `20261009_yamnet_esc50_w8a16_v1`.

**推理覆盖:** ESC-50 全部 2000 条音频、20000 个窗口.

**精度评测:** 未参与校准的第 2-5 折共 1600 条,排除 3 个无直接标签映射类别的 96 条,
按 47 类、1504 条有效样本计算 AP,不是 AudioSet mAP.

表中耗时为该交付运行的 NPU 单次推理平均耗时,一次输入 `1×1×96×64`,batch=1,
排除预热和音频前后处理.两种保留方案及不同运行条件的配对对照见
[YAMNet README](audio/ambient_sound/yamnet/README.md).

Depth Anything V2 Small 的 DA-2K 同协议 ONNX FP32 点对准确率为 **94.83%**,板端为 **85.78%**,精度下降 **9.04 个百分点**。结果见 [summary.json](navigation/single_camera_depth/depth_anything_v2_small/results/summary.json)。

精度变化正数表示改善,负数表示下降: 准确率和 AP/AR 用板端减 ONNX,WER 用 ONNX 减板端。量化方式见各模型 README。ORT `session.Run` 与独立 NPU 调用耗时分开记录,不直接混排。

## 机器人模型推荐路线图

推荐顺序以“补齐机器人能力闭环、复用现有工程资产、控制 MTK 转换风险”为原则，不按模型
热度排序。

| 优先级 | 推荐模型或工作项 | 目标能力 | 建议目录 | 推荐理由 | 进入实现前的关键门槛 |
| --- | --- | --- | --- | --- | --- |
| P0 | DeepLabV3-Plus-MobileNet | 地面、道路、墙体和障碍物语义分割 | `navigation/segmentation/deeplabv3_plus_mobilenet` | 轻量 CNN、固定输入输出，适合先建立机器人可通行区域闭环 | 锁定官方源码、权重、许可证、数据集和类别映射；先检查空洞卷积与 resize 算子 |
| P0 | YOLOv5s → Genio 5100 | 第二平台部署基线 | 复用现有 YOLOv5s 包 | Genio 720 证据最完整，最适合验证框架的平台抽象与兼容性 | 核对 G5100 SDK、编译架构、Runtime、量化和板端部署差异 |
| P1 | Depth Anything V2 Small | 单目相对深度、障碍物远近和空间结构 | `navigation/single_camera_depth/depth_anything_v2_small` | 能把二维检测扩展为空间感知，直接服务移动机器人导航 | 固定输入尺寸；检查 Transformer、插值和量化误差；不得把相对深度直接当作安全米制距离 |
| P1 | MediaPipe Hand Gesture | 手检测、21 点关键点和手势控制 | `interaction/gesture/mediapipe_hand_gesture` | 能形成直观的人机交互 Demo，并补足现有 RTMPose 的手部能力 | 按子模型分别验证检测、关键点和分类；动作控制必须使用多帧确认与超时失效 |
| 已完成 | YAMNet W8A16 | 警报、撞击、玻璃破碎和机械异常声识别 | `audio/ambient_sound/yamnet` | MobileNetV1 架构轻量，可与 Whisper 形成“语音内容 + 环境事件”双通路 | ESC-50 全量实测已完成；后续接入实时麦克风并使用真实机器人环境噪声验证 |
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

阶段 3：MediaPipe Hand Gesture / YAMNet 实时采集应用
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

## 统一运行流程

1. 在模型 `deploy/run.sh` 顶部按五组配置模型、数据、输出与环境.
2. 在编译主机的本模型目录运行 `bash deploy/run.sh`: 转换或准备模型、评测 ONNX 精度、交叉编译 C++ 程序,然后上传.
3. 登录开发板,执行第一步打印的 `[NEXT]` 命令.在部署目录运行 `bash run.sh` 完成测试.
4. 上传最新 `summary.json`,更新本模型 README 的核心指标.量化方式只在 README 说明.

板端统一使用 `models/` 保存模型和参数、`board/` 保存板端程序、评测代码及必要依赖.全量结果写入 `BOARD_RESULTS_DIR/<运行编号>/summary.json`,成功后保留汇总与少量效果示例,失败时保留现场.

各模型在板端直接运行 `bash run.sh` 默认执行全量测试,无需模式参数.Whisper 的双模型以及各任务的数据与指标属于必要差异,详见对应模型 README.

`TFLite/` 是官方模型合集与通用基准工具,采用已有的模型格式,不作为上述单模型移植目录.
各模型 README 统一提供“效果示例”.少量输入随脚本上传,来源与样本清单在编译时从数据集生成且不纳入 Git.测试完成后在本次结果目录生成 `examples/output/`,不保留全量原始预测.将 `summary.json` 与少量效果文件取回本地后更新指标和展示.

## 单模型交付要求

所有模型的原始权重、离线包、转换产物和来源记录统一放在 `models/`,必要的上游源码放在 `models/upstream/`.不再单独设置原始模型目录.

每个模型目录使用以下结构:

```text
<model>/
├── README.md
├── model_card.md
├── LICENSE
├── models/
│   └── source_url.txt
├── deploy/
│   ├── run.sh
│   ├── host/
│   └── board/
├── examples/
│   ├── input/            # 固定示例与来源清单.
│   └── output/           # 实际板端效果.
└── results/summary.json
```

完整交付必须形成以下证据链：

1. 使用官方或原作者源码与权重，并记录版本、下载地址、许可证和文件大小。
2. 保存原始框架推理结果，自行导出 ONNX 或 TFLite，并完成数值一致性比较。
3. 完成 MTK 量化和 NPU 编译，记录工具链、命令、输入输出及量化配置。
4. 在目标开发板执行真实模型推理和可复现 Demo，而不是只检查 Runtime 或模型文件。
5. 在编译主机评测 ONNX 核心精度,与板端同协议精度比较。
6. 汇总板端 NPU 平均耗时、峰值 RSS、核心精度及相对 ONNX 的精度变化。
7. 在 results/summary.json 保留最新测试汇总,更新模型 README、模型卡、示例输出和注册表状态。

对于机器人应用，还应在模型指标之外记录应用层输出，例如可通行区域、目标相对深度、
稳定 `track_id`、手势防抖结果或环境声告警；单张可视化图片不能替代连续运行和安全边界验证。
