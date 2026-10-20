# 模型目录

本目录按照“Scenario → Category → Model”三级结构组织模型，覆盖模型来源、转换、量化、
MTK NPU 部署、板端 Demo、精度与性能评估以及文档交付。Qualcomm AI Hub Models 仅作为
目录结构、模型卡、Demo 和评测方式的 `delivery_reference`，实际模型必须使用官方或原作者
公开源码与权重完成 MTK 侧转换和验证。

## Models List

依据 `docs/MTK_models.xlsx` 的“Effort评估”工作表,按“是否适配”为“做”保留 48 个版本,序号沿用原表.表末同时列出已适配的 YOLOv5s 和 Whisper-Tiny,共 50 行.型号与结果按具体版本对应,不在不同规模模型之间复用数据.具体版本与来源以 `registry/target_models.yaml` 及各模型文档为准.

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
| CenterPoint-Pillar 0.2（双 PFN,10 sweeps）<br>原表 53<br>[tianweiy/CenterPoint](https://github.com/tianweiy/CenterPoint) | 点云 / 点云 | 每点 5 维,10 sweeps<br>参数量待统计 | — | nuScenes,10 类<br>mAP / NDS | — | — | — | — |
| MIT BEVFusion<br>原表 54 | 点云 / 3D融合 | — | — | — | — | — | — | — |
| [YOLOv5s-v7.0](perception/object_detection/yolov5s/README.md)<br>YOLOv5-S,官方 `v7.0` 权重 | 感知 / 目标检测 | RGB `1×3×640×640`<br>7.24 M | 原始 .pt: 14.81<br>ONNX: 28.94 | COCO val2017,5000 张<br>bbox mAP@0.5:0.95<br>ONNX: 37.09% | **W8A8**<br>DLA: 7.66 MB<br>NPU: 9.757 ms/图<br>RSS: 33.23 MiB | 35.86%<br>下降 1.2272 个百分点<br>[报告](perception/object_detection/yolov5s/results/summary.json) | — | — |
| [Whisper-Tiny](audio/stt/whisper_tiny/README.md)<br>OpenAI 多语言 `tiny`,源码 `v20250625` | 音频 / 语音识别 | encoder: Log-Mel `1×80×3000`; decoder: token + KV cache<br>37.18 M | 原始 .pt: 75.57<br>ONNX: 230.79 (双模型合计) | LibriSpeech test-clean,2620 条音频<br>WER (越低越好)<br>ONNX: 7.5584% | **浮点 (实际 W/A 位宽待核实)**<br>DLA: 116.03 MB<br>NPU: 379.187 ms/条音频 (Encoder+Decoder)<br>RSS: 105.53 MiB | 7.5603%<br>上升 0.0019 个百分点<br>[报告](audio/stt/whisper_tiny/results/summary.json) | — | — |

### 数据说明

- 输入按 `N×C×H×W` 表示,图像 batch=1;音频采样率为 16 kHz.参数量 `1 M = 100 万`,不含 BatchNorm 运行统计量等缓冲区.
- 文件大小采用十进制 MB,`1 MB = 1,000,000 bytes`;RSS 为 MiB,与文件大小分开记录.原始权重为当前官方/原作者检查点文件实际大小,保留格式,不将其等同于 FP32 网络大小.多分片、双模型按实际必需文件合计;DLA 不含外置 embedding,有外置权重时另列部署合计.
- `WnAm` 表示权重 n 位、激活 m 位.浮点必须同时标明 FP16/FP32,不可与同位宽整数定点量化混淆;W8A8 为 INT8 PTQ,主要权重采用逐输出通道量化,不表示偏置、全部算子或文件 I/O 都是 8 位.Whisper 已确认浮点部署,现有报告未明确全部 W/A 位宽,暂不强行标为 W16A16.
- 精度仅比较同模型、同数据集与同评测协议的参考基线.表中直接写“上升/下降”,准确率/AP/AR 越高越好,WER/PPL 越低越好;PPL 差值不是百分点.差值按报告原始数值计算,不由展示后舍入值相减.MiniCPM5 使用原框架 FP32 参考,不是 ONNX 基线.
- 指标来源为各模型结果报告.不同任务的加载、计时和内存范围不同,不作为统一性能排名.


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

## 使用与资源

各模型的 README 提供配置、编译上传及板端运行方法.公共环境、NAS 资源规则和新增模型要求见[仓库说明](../README.md),板端目录及评价要求见[Genio 720 规范](../docs/genio_720.md).

型号来源由 `registry/target_models.yaml` 和各模型的 `models/source_url.txt` 记录.原始权重、转换产物和数据集保存在 NAS,不放进代码目录或临时目录作为正式资源.
