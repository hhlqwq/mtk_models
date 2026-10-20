# 模型目录

本目录按照“Scenario → Category → Model”三级结构组织模型，覆盖模型来源、转换、量化、
MTK NPU 部署、板端 Demo、精度与性能评估以及文档交付。Qualcomm AI Hub Models 仅作为
目录结构、模型卡、Demo 和评测方式的 `delivery_reference`，实际模型必须使用官方或原作者
公开源码与权重完成 MTK 侧转换和验证。

## 模型规格

本次补充 LLM: [DeepSeek-R1-Distill-Qwen-1.5B](gen_ai/llm/deepseek_r1_distill_qwen_1_5b/README.md).
目标为中英文文本问答与推理,来源为 DeepSeek 官方权重,地瓜 RDK 仅作为部署参考.
Genio 720 已完成 FP16 双语 Demo 与三后端样例数值检查 (`board_verified`),
Genio 5100 尚未开始.原生 W4A16 双语推理已验证,Decode 约 16.75 Token/s,资源约 2.04 GB.
英文输出存在语法问题,正式量化质量待评测; 详见模型 README 中的独立原生报告.

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

每个已部署模型的汇总必须保留 `model_size`,单位为 MB (十进制),保留两位小数.
`compiled_dla` 为本次使用的 DLA 文件合计,`auxiliary_weights` 为外置 embedding,
`deployment_total` 为二者合计,`files` 保存文件明细.Whisper 的 encoder/decoder、
LLM 的全部实际使用分片须合计,不包含历史副本、评测输入输出或运行时 RSS.
`source_weights`、`fp32_onnx`、`quantized_tflite` 没有实际文件时为 `null`.
现有大小记录直接取自板端现存文件,没有重新执行推理或评测; 采集日期与原评测运行编号分别保留.
原生 LLM 大小同时保存在 `native_summary.json`,与 FP16 部署分开记录.
后续部署脚本通过 `tools/model_size.py` 自动补充大小,新模型须按模板接入该入口.

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

## Models List

清单依据 `docs/MTK_models.xlsx` 的“Effort评估”工作表,按“是否适配”列为“做”筛选,共 48 个版本条目.序号沿用原表,表示本轮适配范围,不代表已完成部署.

| 原表序号 | 场景 | 类别 | 具体版本 |
| ---: | --- | --- | --- |
| 1 | 感知 | 目标检测 | YOLOv5n-v7.0（640×640） |
| 2 | 感知 | 目标检测 | YOLOv5m-v7.0（640×640） |
| 3 | 感知 | 目标检测 | YOLOv8n（640×640） |
| 4 | 感知 | 目标检测 | YOLOv8m（640×640） |
| 5 | 感知 | 目标检测 | RT-DETRv2-S（640×640） |
| 6 | 感知 | 分类 | SqueezeNet-1.1 |
| 7 | 感知 | 分类 | ResNet50 |
| 8 | 感知 | 分类 | ResNet101 |
| 9 | 感知 | 分类 | MobileNetV3-Large |
| 10 | 感知 | 分类 | ViT-B/16 |
| 11 | 交互 | 手势识别 | MediaPipe Gesture Recognizer（Palm+Hand Landmark+Gesture Classifier） |
| 12 | 交互 | 手部关键点 | DepthAI Hand Tracker（PalmDetection+HandLandmark 21点） |
| 13 | 交互 | 人脸识别 | MobileFaceNet |
| 14 | 交互 | 姿态估计 | MoveNet SinglePose Lightning（192×192） |
| 15 | 交互 | 姿态估计 | RTMPose-m WholeBody（256×192，133点） |
| 16 | 交互 | 姿态估计 | HRNet-W32（COCO 17点） |
| 17 | 交互 | 姿态估计 | BlazePose GHUM Lite（33点） |
| 18 | 交互 | 文本识别 | PROCR v4 |
| 19 | 导航 | 语义分割 | Fast-SCNN（Cityscapes 19类） |
| 20 | 导航 | 语义分割 | DeepLabV3-MobileNetV3-Large（VOC 21类） |
| 21 | 导航 | 语义分割 | segformer-b0-finetuned-ade-512-512 |
| 22 | 导航 | 语义分割 | FastSAM-s |
| 23 | 导航 | 语义分割 | FastSAM-x |
| 24 | 导航 | 语义分割 | EdgeSAM（Image Encoder+Prompt Decoder） |
| 25 | 导航 | 单目深度 | Depth Anything V2-Small（ViT-S） |
| 26 | 导航 | 双目深度 | StereoNetPlus-v1.2.1（MixVarGENet + FPN，SceneFlow） |
| 27 | 导航 | 视觉基础模型 | DINOv2 ViT-S/14（dinov2_vits14） |
| 28 | 生成式AI | 视觉语言 | CLIP ViT-B/16 |
| 31 | 生成式AI | 大语言模型 | MiniCPM5-2B |
| 32 | 生成式AI | 大语言模型 | DeepSeek-R1-Distill-Qwen-1.5B |
| 33 | 生成式AI | 大语言模型 | DeepSeek-R1-Distill-Qwen-7B |
| 34 | 生成式AI | VLM | InternVL3-1B |
| 35 | 生成式AI | VLM | InternVL3-8B |
| 37 | 生成式AI | VLM | Gemma 4 E2B |
| 38 | 生成式AI | VLM | Gemma 4 E4B |
| 42 | 音频 | 语音合成 | MeloTTS |
| 43 | 音频 | 语音合成 | PiperTTS |
| 44 | 音频 | 环境音频 | WebRTC AEC3 |
| 45 | 音频 | 环境音频 | DeepFilterNet v3 |
| 46 | 音频 | 环境音频 | YAMNet（TF Hub，521类） |
| 47 | SLAM | 目标跟踪 | YOLOv8n + OSNet_x0_25 + BoT-SORT（BoxMOT） |
| 48 | SLAM | 定位 | MobileNetVLAD（mobilenetvlad_depth-0.35） |
| 49 | SLAM | 定位 | HF-Net |
| 50 | SLAM | 定位 | SuperPoint（MagicLeap pretrained） |
| 51 | SLAM | 定位 | LightGlue（SuperPoint，固定1024关键点） |
| 52 | 点云 | 点云 | PointPillar-Tiny |
| 53 | 点云 | 点云 | CenterPoint-Pillar 0.2（SECOND + SECFPN） |
| 54 | 点云 | 3D融合 | MIT BEVFusion |

原表标记“不做”的 6 项: Llama3.2, Qwen3, Qwen3-VL, whisper, Qwen3-ASR, Moonshine tiny (streaming).此筛选不删除仓库已有模型或历史实测结果.

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

[MiniCPM5-2B](gen_ai/llm/minicpm5_2b/README.md) 已完成轻量 Hessian 全 W4A16 双 DLA、板端性能和首轮同协议精度对比.固定 1000 题 MMLU 0-shot: FP32 55.9%,NPU 41.3%;WikiText2 test 前 128 行 PPL: FP32 40.04,NPU 48.38.精度退化明显,保持未通过质量验收状态.

LLM 的 Prefill 与 Decode 各一个 DLA,速度统一 token/s.本次 MiniCPM5 固定 128 输入、最多 128 输出的 Prefill 288.67 token/s,Decode 8.15 token/s.使用 `tools/llm/` 公共工具;首轮子集、四块诊断和完整测试集分别记录,不混用范围.16 个训练校准样本的 Hessian 修复已完成首轮实测,累计层误差优化关闭,详见模型 README.

## 外部模型资源

此处是模型代码目录.模型二进制与配套运行参数保存在本地配置指定的外部存储中,映射为 `models_root/场景/类别/模型名/`.不在 Git 文件中写入内部完整路径.
