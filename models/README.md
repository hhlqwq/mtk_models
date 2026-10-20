# 模型目录

本目录按照“Scenario → Category → Model”三级结构组织模型，覆盖模型来源、转换、量化、
MTK NPU 部署、板端 Demo、精度与性能评估以及文档交付。Qualcomm AI Hub Models 仅作为
目录结构、模型卡、Demo 和评测方式的 `delivery_reference`，实际模型必须使用官方或原作者
公开源码与权重完成 MTK 侧转换和验证。

## Models List

依据 `docs/MTK_models.xlsx` 的“Effort评估”工作表,按“是否适配”为“做”保留 48 个版本,序号沿用原表.另在表末保留本轮计划外的 YOLOv5s 和 Whisper-Tiny 历史实测,共 50 行.型号与结果按具体版本对应,不在不同规模模型之间复用数据.

原始权重与 ONNX 为公共参考;量化方式、编译后大小、性能和精度按平台分别记录.G5100 暂无已回收实测,预留独立列.`—` 表示尚无可引用记录,不代表零或已完成适配.点击模型名称查看完整版本、来源和限制,点击“报告”查看实测依据.平台“编译与性能”单元格依次列出量化方式、DLA 大小、NPU 耗时和 RSS.

| 模型 / 版本 (原表序号) | 场景 / 类别 | 输入 / 参数量 (M) | 原始权重 / FP32 ONNX (MB) | 数据集 / 指标 / 参考基线 | G720 量化方式 / 编译与性能 | G720 精度 / 变化 | G5100 量化方式 / 编译与性能 | G5100 精度 / 变化 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5n-v7.0（640×640）<br>原表 1 | 感知 / 目标检测 | — | — | — | — | — | — | — |
| YOLOv5m-v7.0（640×640）<br>原表 2 | 感知 / 目标检测 | — | — | — | — | — | — | — |
| [YOLOv8n（640×640）](perception/object_detection/yolov8n/README.md)<br>原表 3<br>YOLOv8-N,Ultralytics `v8.0.111` | 感知 / 目标检测 | RGB `1×3×640×640`<br>3.16 M | 原始 .pt: 6.53<br>ONNX: 12.66 | COCO val2017,5000 张<br>bbox mAP@0.5:0.95<br>ONNX: 36.6260% | **W8A8**<br>DLA: 3.53 MB<br>NPU: 6.283 ms/图<br>RSS: 31.32 MiB | 35.3510%<br>下降 1.2750 个百分点<br>[报告](perception/object_detection/yolov8n/results/summary.json) | — | — |
| YOLOv8m（640×640）<br>原表 4 | 感知 / 目标检测 | — | — | — | — | — | — | — |
| RT-DETRv2-S（640×640）<br>原表 5 | 感知 / 目标检测 | — | — | — | — | — | — | — |
| SqueezeNet-1.1<br>原表 6 | 感知 / 分类 | — | — | — | — | — | — | — |
| ResNet50<br>原表 7 | 感知 / 分类 | — | — | — | — | — | — | — |
| ResNet101<br>原表 8 | 感知 / 分类 | — | — | — | — | — | — | — |
| MobileNetV3-Large<br>原表 9 | 感知 / 分类 | — | — | — | — | — | — | — |
| [ViT-B/16](perception/image_classification/vit_base_patch16_224/README.md)<br>原表 10<br>ViT-B/16,TorchVision `IMAGENET1K_V1` | 感知 / 分类 | RGB `1×3×224×224`<br>86.57 M | 原始 .pth: 346.33<br>ONNX: 346.44 | ImageNet val2012,50000 张<br>Top-1 Accuracy<br>ONNX: 80.66% | **W8A8**<br>DLA: 88.00 MB<br>NPU: 51.754 ms/图<br>RSS: 141.30 MiB | 79.40%<br>下降 1.2660 个百分点<br>[报告](perception/image_classification/vit_base_patch16_224/results/summary.json) | — | — |
| MediaPipe Gesture Recognizer（Palm+Hand Landmark+Gesture Classifier）<br>原表 11 | 交互 / 手势识别 | — | — | — | — | — | — | — |
| DepthAI Hand Tracker（PalmDetection+HandLandmark 21点）<br>原表 12 | 交互 / 手部关键点 | — | — | — | — | — | — | — |
| [MobileFaceNet](interaction/face_recognition/mobilefacenet/README.md)<br>原表 13<br>foamliu MobileFaceNet,`v1.0` 权重,128 维特征 | 交互 / 人脸识别 | RGB 对齐人脸 `1×3×112×112`<br>1.00 M | 原始 .pt: 4.14<br>ONNX: 4.00 | LFW,MTCNN 五点对齐,6000 对<br>验证准确率<br>ONNX: 99.38% | **W8A8**<br>DLA: 1.41 MB<br>NPU: 0.515 ms/图<br>RSS: 8.25 MiB | 99.32%<br>下降 0.0667 个百分点<br>[报告](interaction/face_recognition/mobilefacenet/results/summary.json) | — | — |
| MoveNet SinglePose Lightning（192×192）<br>原表 14 | 交互 / 姿态估计 | — | — | — | — | — | — | — |
| [RTMPose-m WholeBody（256×192，133点）](interaction/pose_detection/rtmpose_body2d/README.md)<br>原表 15<br>RTMPose-M WholeBody 133 点,MMPose `v1.3.2` | 交互 / 姿态估计 | RGB 人体裁剪 `1×3×256×192`<br>17.95 M | 原始 .pth: 72.01<br>ONNX: 71.83 | COCO-WholeBody val2017,104125 个人体<br>WholeBody AP<br>ONNX: 57.04% | **W8A8**<br>DLA: 19.14 MB<br>NPU: 3.862 ms/人体裁剪<br>RSS: 32.74 MiB | 53.24%<br>下降 3.7961 个百分点<br>[报告](interaction/pose_detection/rtmpose_body2d/results/summary.json) | — | — |
| HRNet-W32（COCO 17点）<br>原表 16 | 交互 / 姿态估计 | — | — | — | — | — | — | — |
| BlazePose GHUM Lite（33点）<br>原表 17 | 交互 / 姿态估计 | — | — | — | — | — | — | — |
| PROCR v4<br>原表 18 | 交互 / 文本识别 | — | — | — | — | — | — | — |
| Fast-SCNN（Cityscapes 19类）<br>原表 19 | 导航 / 语义分割 | — | — | — | — | — | — | — |
| DeepLabV3-MobileNetV3-Large（VOC 21类）<br>原表 20 | 导航 / 语义分割 | — | — | — | — | — | — | — |
| segformer-b0-finetuned-ade-512-512<br>原表 21 | 导航 / 语义分割 | — | — | — | — | — | — | — |
| [FastSAM-s](navigation/segmentation/fastsam/README.md)<br>原表 22<br>官方 `FastSAM-s.pt`,Ultralytics `v8.0.111` | 导航 / 语义分割 | RGB `1×3×640×640`<br>11.79 M | 原始 .pt: 23.83<br>ONNX: 47.18 | COCO val2017,5000 张<br>类别无关 segm AR@100<br>ONNX: 39.02% | **W8A8**<br>DLA: 12.29 MB<br>NPU: 14.725 ms/图<br>RSS: 64.55 MiB | 37.57%<br>下降 1.4507 个百分点<br>[报告](navigation/segmentation/fastsam/results/summary.json) | — | — |
| FastSAM-x<br>原表 23 | 导航 / 语义分割 | — | — | — | — | — | — | — |
| EdgeSAM（Image Encoder+Prompt Decoder）<br>原表 24 | 导航 / 语义分割 | — | — | — | — | — | — | — |
| [Depth Anything V2-Small（ViT-S）](navigation/single_camera_depth/depth_anything_v2_small/README.md)<br>原表 25<br>Depth Anything V2 Small,ViT-S/14 | 导航 / 单目深度 | RGB `1×3×518×518`<br>24.79 M | 原始 .pth: 99.22<br>ONNX: 98.97 | DA-2K,2068 个点对<br>点对准确率<br>ONNX: 94.83% | **W8A8**<br>DLA: 25.83 MB<br>NPU: 133.017 ms/图<br>RSS: 33.04 MiB | 85.78%<br>下降 9.0426 个百分点<br>[报告](navigation/single_camera_depth/depth_anything_v2_small/results/summary.json) | — | — |
| StereoNetPlus-v1.2.1（MixVarGENet + FPN，SceneFlow）<br>原表 26 | 导航 / 双目深度 | — | — | — | — | — | — | — |
| DINOv2 ViT-S/14（dinov2_vits14）<br>原表 27 | 导航 / 视觉基础模型 | — | — | — | — | — | — | — |
| CLIP ViT-B/16<br>原表 28 | 生成式AI / 视觉语言 | — | — | — | — | — | — | — |
| [MiniCPM5-2B](gen_ai/llm/minicpm5_2b/README.md)<br>原表 31 | 生成式AI / 大语言模型 | Token + KV Cache<br>上下文 1024,Prefill 128 / Decode 1<br>约 2520 M | 原始 .safetensors: 5033.56<br>ONNX: — | MMLU test,57 学科固定 1000 题,0-shot<br>Accuracy,FP32: 55.90%<br>WikiText2 test 前 128 行/7366 Token<br>PPL,FP32: 40.04 (均为子集) | **W4A16 (INT4/INT16)**<br>DLA: 2292.52 MB<br>embedding: 534.77 MB<br>部署合计: 2827.29 MB<br>NPU: Prefill: 288.67 Token/s<br>Decode: 8.15 Token/s (128 输入/最多 128 输出)<br>RSS: — | MMLU: 41.30%,下降 14.60 个百分点<br>WikiText2 PPL: 48.38,上升 8.34<br>首轮子集,质量未通过<br>[报告](gen_ai/llm/minicpm5_2b/results/summary.json) | — | — |
| [DeepSeek-R1-Distill-Qwen-1.5B](gen_ai/llm/deepseek_r1_distill_qwen_1_5b/README.md)<br>原表 32<br>DeepSeek 官方固定 revision,实际参数包含独立 LM Head | 生成式AI / 大语言模型 | Token + KV Cache,上下文 1024,28 层<br>1777.09 M | 原始 .safetensors: 3554.21<br>ONNX: 6175.24 (38 个分片合计) | 自编双语短文本,26 Token (非正式基准)<br>PPL,ONNX: 144.57 | **W16A16 (FP16 浮点)**<br>DLA: 3089.46 MB<br>embedding: 466.75 MB<br>部署合计: 3556.21 MB<br>NPU: 220.11 ms/Token (纯 NPU)<br>RSS: 148.10 MiB | PPL: 145.05,上升 0.48<br>仅样例数值检查,非正式精度<br>[报告](gen_ai/llm/deepseek_r1_distill_qwen_1_5b/results/summary.json) | — | — |
| DeepSeek-R1-Distill-Qwen-7B<br>原表 33 | 生成式AI / 大语言模型 | — | — | — | — | — | — | — |
| InternVL3-1B<br>原表 34 | 生成式AI / VLM | — | — | — | — | — | — | — |
| InternVL3-8B<br>原表 35 | 生成式AI / VLM | — | — | — | — | — | — | — |
| Gemma 4 E2B<br>原表 37 | 生成式AI / VLM | — | — | — | — | — | — | — |
| Gemma 4 E4B<br>原表 38 | 生成式AI / VLM | — | — | — | — | — | — | — |
| MeloTTS<br>原表 42 | 音频 / 语音合成 | — | — | — | — | — | — | — |
| PiperTTS<br>原表 43 | 音频 / 语音合成 | — | — | — | — | — | — | — |
| WebRTC AEC3<br>原表 44 | 音频 / 环境音频 | — | — | — | — | — | — | — |
| DeepFilterNet v3<br>原表 45 | 音频 / 环境音频 | — | — | — | — | — | — | — |
| [YAMNet（TF Hub，521类）](audio/ambient_sound/yamnet/README.md)<br>原表 46<br>Google AudioSet 521 类,源码 `34a2132` | 音频 / 环境音频 | Log-Mel `1×1×96×64`,单声道音频<br>3.73 M | 原始 .h5: 15.30<br>ONNX: 14.94 | ESC-50 第 2-5 折,47 类/1504 条有效音频<br>标签投影宏平均 AP<br>ONNX: 74.1456% | **W8A16**<br>DLA: 3.99 MB<br>NPU: 0.516 ms/窗口<br>RSS: 10.75 MiB | 73.8964%<br>下降 0.2491 个百分点<br>[报告](audio/ambient_sound/yamnet/results/summary.json) | — | — |
| YOLOv8n + OSNet_x0_25 + BoT-SORT（BoxMOT）<br>原表 47 | SLAM / 目标跟踪 | — | — | — | — | — | — | — |
| MobileNetVLAD（mobilenetvlad_depth-0.35）<br>原表 48 | SLAM / 定位 | — | — | — | — | — | — | — |
| HF-Net<br>原表 49 | SLAM / 定位 | — | — | — | — | — | — | — |
| SuperPoint（MagicLeap pretrained）<br>原表 50 | SLAM / 定位 | — | — | — | — | — | — | — |
| LightGlue（SuperPoint，固定1024关键点）<br>原表 51 | SLAM / 定位 | — | — | — | — | — | — | — |
| PointPillar-Tiny<br>原表 52 | 点云 / 点云 | — | — | — | — | — | — | — |
| CenterPoint-Pillar 0.2（SECOND + SECFPN）<br>原表 53 | 点云 / 点云 | — | — | — | — | — | — | — |
| MIT BEVFusion<br>原表 54 | 点云 / 3D融合 | — | — | — | — | — | — | — |
| [YOLOv5s-v7.0 (本轮计划外)](perception/object_detection/yolov5s/README.md)<br>YOLOv5-S,官方 `v7.0` 权重 | 感知 / 目标检测 | RGB `1×3×640×640`<br>7.24 M | 原始 .pt: 14.81<br>ONNX: 28.94 | COCO val2017,5000 张<br>bbox mAP@0.5:0.95<br>ONNX: 37.09% | **W8A8**<br>DLA: 7.66 MB<br>NPU: 9.757 ms/图<br>RSS: 33.23 MiB | 35.86%<br>下降 1.2272 个百分点<br>[报告](perception/object_detection/yolov5s/results/summary.json) | — | — |
| [Whisper-Tiny (本轮计划外)](audio/stt/whisper_tiny/README.md)<br>OpenAI 多语言 `tiny`,源码 `v20250625` | 音频 / 语音识别 | encoder: Log-Mel `1×80×3000`; decoder: token + KV cache<br>37.18 M | 原始 .pt: 75.57<br>ONNX: 230.79 (双模型合计) | LibriSpeech test-clean,2620 条音频<br>WER (越低越好)<br>ONNX: 7.5584% | **浮点 (实际 W/A 位宽待核实)**<br>DLA: 116.03 MB<br>NPU: 379.187 ms/条音频 (Encoder+Decoder)<br>RSS: 105.53 MiB | 7.5603%<br>上升 0.0019 个百分点<br>[报告](audio/stt/whisper_tiny/results/summary.json) | — | — |

### 数据说明

- 输入按 `N×C×H×W` 表示,图像 batch=1;音频采样率为 16 kHz.参数量 `1 M = 100 万`,不含 BatchNorm 运行统计量等缓冲区.
- 文件大小采用十进制 MB,`1 MB = 1,000,000 bytes`;RSS 为 MiB,与文件大小分开记录.原始权重为当前官方/原作者检查点文件实际大小,保留格式,不将其等同于 FP32 网络大小.多分片、双模型按实际必需文件合计;DLA 不含外置 embedding,有外置权重时另列部署合计.
- `WnAm` 表示权重 n 位、激活 m 位.浮点必须同时标明 FP16/FP32,不可与同位宽整数定点量化混淆;W8A8 为 INT8 PTQ,主要权重采用逐输出通道量化,不表示偏置、全部算子或文件 I/O 都是 8 位.Whisper 已确认浮点部署,现有报告未明确全部 W/A 位宽,暂不强行标为 W16A16.
- 精度仅比较同模型、同数据集与同评测协议的参考基线.表中直接写“上升/下降”,准确率/AP/AR 越高越好,WER/PPL 越低越好;PPL 差值不是百分点.差值按报告原始数值计算,不由展示后舍入值相减.MiniCPM5 使用原框架 FP32 参考,不是 ONNX 基线.
- 原始权重大小为本次只读文件统计;性能、RSS、精度与 DLA 大小沿用已归档结果,本次未重新运行模型.不同模型的常驻/逐图加载、计时和内存范围不同,不直接作为统一性能排名.
- Whisper 完整原始网络为 37.18 M;encoder/decoder ONNX 分别 32.87/197.92 MB,DLA 分别 16.60/99.43 MB.完整来源见各模型 `model_card.md` 与 `models/source_url.txt`.

原表标记“不做”的 6 项为 Llama3.2、Qwen3、Qwen3-VL、whisper、Qwen3-ASR、Moonshine tiny (streaming),不删除已有历史结果.

### 评测口径与限制

DeepSeek 结果来自运行 `20261010_deepseek_fp16_v3`,batch=1、上下文 1024、FP16、
逐 Token Prefill、Greedy 生成上限 512.中文 / 英文 Demo 分别在 73 / 412 Token 到达 EOS,
TTFT 为 3.245 / 3.112 秒,端到端 Decode 为 3.78 / 3.81 Token/s.
表中 220.11 ms 为模型算子的纯 NPU 调用平均耗时; RSS 未覆盖全部 NPU 驱动内存,
本次系统可用内存下降约 3512.09 MiB.三后端自编短文本 PPL 为 144.57 / 144.57 / 145.05,
仅验证数值,不能替代正式基准.结果与实际文本见 [DeepSeek README](gen_ai/llm/deepseek_r1_distill_qwen_1_5b/README.md).

DeepSeek 另有独立的 **W4A16 (INT4/INT16)** 原生方案,DLA 合计 **1570.21 MB**,embedding **466.75 MB**,部署合计 **2036.96 MB**;双语样例 Decode 约 **16.75 Token/s**.该方案尚未完成正式精度评测,英文样例存在语法问题,不得套用表中 FP16 的 PPL 或 RSS.详见 [native_summary.json](gen_ai/llm/deepseek_r1_distill_qwen_1_5b/results/native_summary.json).

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
