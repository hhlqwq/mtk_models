# 模型目录

本目录按照“Scenario → Category → Model”三级结构组织模型，覆盖模型来源、转换、量化、
MTK NPU 部署、板端 Demo、精度与性能评估以及文档交付。Qualcomm AI Hub Models 仅作为
目录结构、模型卡、Demo 和评测方式的 `delivery_reference`，实际模型必须使用官方或原作者
公开源码与权重完成 MTK 侧转换和验证。

## Models List

列出计划适配的模型及已完成量化的版本.型号与结果按具体版本对应,不在不同规模模型之间复用数据.具体版本与来源见各模型文档.

原始权重与 ONNX 为公共参考;量化方式、编译后大小、性能和精度按平台分别记录.G5100 暂无已回收实测,预留独立列.`—` 表示尚无可引用记录,不代表零或已完成适配.点击模型名称查看完整版本、来源和限制,点击“报告”查看实测依据.平台“编译与性能”单元格依次列出量化方式、DLA 大小、NPU 耗时和 RSS.

| 模型 / 版本 | 场景 / 类别 | 输入 / 参数量 | 原始权重 / FP32 ONNX | 数据集 / 指标 / 参考基线 | G720 量化方式 / 编译与性能 | G720 精度 / 变化 | G5100 量化方式 / 编译与性能 | G5100 精度 / 变化 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5n-v7.0（640×640） | 感知 / 目标检测 | — | — | — | — | — | — | — |
| YOLOv5m-v7.0（640×640） | 感知 / 目标检测 | — | — | — | — | — | — | — |
| [YOLOv8n（640×640）](perception/object_detection/yolov8n/README.md)<br>YOLOv8-N,Ultralytics `v8.0.111` | 感知 / 目标检测 | RGB `1×3×640×640`<br>3.16 M | 原始: 6.53 MB<br>ONNX: 12.66 MB | COCO val2017,5000 张<br>bbox mAP@0.5:0.95<br>ONNX: 36.63% | **W8A8**<br>DLA: 3.53 MB<br>NPU: 6.283 ms<br>RSS: 31.32 MiB | 35.35%<br>下降 1.28 个百分点<br>[报告](perception/object_detection/yolov8n/results/summary.json) | — | — |
| YOLOv8m（640×640） | 感知 / 目标检测 | — | — | — | — | — | — | — |
| RT-DETRv2-S（640×640） | 感知 / 目标检测 | — | — | — | — | — | — | — |
| SqueezeNet-1.1 | 感知 / 分类 | — | — | — | — | — | — | — |
| ResNet50 | 感知 / 分类 | — | — | — | — | — | — | — |
| ResNet101 | 感知 / 分类 | — | — | — | — | — | — | — |
| MobileNetV3-Large | 感知 / 分类 | — | — | — | — | — | — | — |
| [ViT-B/16](perception/image_classification/vit_base_patch16_224/README.md)<br>ViT-B/16,TorchVision `IMAGENET1K_V1` | 感知 / 分类 | RGB `1×3×224×224`<br>86.57 M | 原始: 346.33 MB<br>ONNX: 346.44 MB | ImageNet val2012,50000 张<br>Top-1 Accuracy<br>ONNX: 80.66% | **W8A8**<br>DLA: 88.00 MB<br>NPU: 51.754 ms<br>RSS: 141.30 MiB | 79.40%<br>下降 1.27 个百分点<br>[报告](perception/image_classification/vit_base_patch16_224/results/summary.json) | — | — |
| MediaPipe Gesture Recognizer（Palm+Hand Landmark+Gesture Classifier） | 交互 / 手势识别 | — | — | — | — | — | — | — |
| DepthAI Hand Tracker（PalmDetection+HandLandmark 21点） | 交互 / 手部关键点 | — | — | — | — | — | — | — |
| [MobileFaceNet](interaction/face_recognition/mobilefacenet/README.md)<br>foamliu MobileFaceNet,`v1.0` 权重,128 维特征 | 交互 / 人脸识别 | RGB 对齐人脸 `1×3×112×112`<br>1.00 M | 原始: 4.14 MB<br>ONNX: 4.00 MB | LFW,MTCNN 五点对齐,6000 对<br>验证准确率<br>ONNX: 99.38% | **W8A8**<br>DLA: 1.41 MB<br>NPU: 0.515 ms<br>RSS: 8.25 MiB | 99.32%<br>下降 0.07 个百分点<br>[报告](interaction/face_recognition/mobilefacenet/results/summary.json) | — | — |
| MoveNet SinglePose Lightning（192×192） | 交互 / 姿态估计 | — | — | — | — | — | — | — |
| [RTMPose-m WholeBody（256×192，133点）](interaction/pose_detection/rtmpose_body2d/README.md)<br>RTMPose-M WholeBody 133 点,MMPose `v1.3.2` | 交互 / 姿态估计 | RGB 人体裁剪 `1×3×256×192`<br>17.95 M | 原始: 72.01 MB<br>ONNX: 71.83 MB | COCO-WholeBody val2017,104125 个人体<br>WholeBody AP<br>ONNX: 57.04% | **W8A8**<br>DLA: 19.14 MB<br>NPU: 3.862 ms<br>RSS: 32.74 MiB | 53.24%<br>下降 3.80 个百分点<br>[报告](interaction/pose_detection/rtmpose_body2d/results/summary.json) | — | — |
| HRNet-W32（COCO 17点） | 交互 / 姿态估计 | — | — | — | — | — | — | — |
| BlazePose GHUM Lite（33点） | 交互 / 姿态估计 | — | — | — | — | — | — | — |
| PROCR v4 | 交互 / 文本识别 | — | — | — | — | — | — | — |
| Fast-SCNN（Cityscapes 19类） | 导航 / 语义分割 | — | — | — | — | — | — | — |
| DeepLabV3-MobileNetV3-Large（VOC 21类） | 导航 / 语义分割 | — | — | — | — | — | — | — |
| segformer-b0-finetuned-ade-512-512 | 导航 / 语义分割 | — | — | — | — | — | — | — |
| [FastSAM-s](navigation/segmentation/fastsam/README.md)<br>官方 `FastSAM-s.pt`,Ultralytics `v8.0.111` | 导航 / 语义分割 | RGB `1×3×640×640`<br>11.79 M | 原始: 23.83 MB<br>ONNX: 47.18 MB | COCO val2017,5000 张<br>类别无关 segm AR@100<br>ONNX: 39.02% | **W8A8**<br>DLA: 12.29 MB<br>NPU: 14.725 ms<br>RSS: 64.55 MiB | 37.57%<br>下降 1.45 个百分点<br>[报告](navigation/segmentation/fastsam/results/summary.json) | — | — |
| FastSAM-x | 导航 / 语义分割 | — | — | — | — | — | — | — |
| EdgeSAM（Image Encoder+Prompt Decoder） | 导航 / 语义分割 | — | — | — | — | — | — | — |
| [Depth Anything V2-Small（ViT-S）](navigation/single_camera_depth/depth_anything_v2_small/README.md)<br>Depth Anything V2 Small,ViT-S/14 | 导航 / 单目深度 | RGB `1×3×518×518`<br>24.79 M | 原始: 99.22 MB<br>ONNX: 98.97 MB | DA-2K,2068 个点对<br>点对准确率<br>ONNX: 94.83% | **W8A8**<br>DLA: 25.83 MB<br>NPU: 133.017 ms<br>RSS: 33.04 MiB | 85.78%<br>下降 9.04 个百分点<br>[报告](navigation/single_camera_depth/depth_anything_v2_small/results/summary.json) | — | — |
| StereoNetPlus-v1.2.1（MixVarGENet + FPN，SceneFlow） | 导航 / 双目深度 | — | — | — | — | — | — | — |
| DINOv2 ViT-S/14（dinov2_vits14） | 导航 / 视觉基础模型 | — | — | — | — | — | — | — |
| CLIP ViT-B/16 | 生成式AI / 视觉语言 | — | — | — | — | — | — | — |
| [MiniCPM5-2B](gen_ai/llm/minicpm5_2b/README.md) | 生成式AI / 大语言模型 | Token + KV Cache<br>上下文 1024 Token<br>Prefill 每批 128 Token<br>逐 Token 解码<br>约 2520 M | 原始: 5033.56 MB | MMLU test,57 学科固定 1000 题,0-shot<br>Accuracy,FP32: 55.90%<br>WikiText2 test 前 128 行/7366 Token<br>PPL,FP32: 40.04 (均为子集) | **W4A16 (INT4/INT16)**<br>DLA: 2292.52 MB<br>embedding: 534.77 MB<br>部署合计: 2827.29 MB<br>Prefill: 288.67 Token/s<br>Decode: 8.15 Token/s<br>RSS: 1152.42 MiB | MMLU: 41.30%,下降 14.60 个百分点<br>WikiText2 PPL: 48.38,上升 8.34<br>固定子集<br>[报告](gen_ai/llm/minicpm5_2b/results/summary.json) | — | — |
| [DeepSeek-R1-Distill-Qwen-1.5B（W4A16）](gen_ai/llm/deepseek_r1_distill_qwen_1_5b/README.md) | 生成式AI / 大语言模型 | Token + KV Cache<br>上下文 1024 Token<br>Prefill 每批 128 Token<br>逐 Token 解码<br>1777.09 M | 原始: 3554.21 MB | 中英文生成样例<br>未进行同协议正式精度对比 | **W4A16 (INT4/INT16)**<br>DLA: 1570.21 MB<br>embedding: 466.75 MB<br>部署合计: 2036.96 MB<br>Decode: 16.75 Token/s<br>RSS: 816.73 MiB | —<br>[报告](gen_ai/llm/deepseek_r1_distill_qwen_1_5b/results/native_summary.json) | — | — |
| DeepSeek-R1-Distill-Qwen-7B | 生成式AI / 大语言模型 | — | — | — | — | — | — | — |
| InternVL3-1B | 生成式AI / VLM | — | — | — | — | — | — | — |
| InternVL3-8B | 生成式AI / VLM | — | — | — | — | — | — | — |
| Gemma 4 E2B | 生成式AI / VLM | — | — | — | — | — | — | — |
| Gemma 4 E4B | 生成式AI / VLM | — | — | — | — | — | — | — |
| MeloTTS | 音频 / 语音合成 | — | — | — | — | — | — | — |
| PiperTTS | 音频 / 语音合成 | — | — | — | — | — | — | — |
| WebRTC AEC3 | 音频 / 环境音频 | — | — | — | — | — | — | — |
| DeepFilterNet v3 | 音频 / 环境音频 | — | — | — | — | — | — | — |
| [YAMNet（TF Hub，521类）](audio/ambient_sound/yamnet/README.md)<br>Google AudioSet 521 类 | 音频 / 环境音频 | Log-Mel `1×1×96×64`,单声道音频<br>3.73 M | 原始: 15.30 MB<br>ONNX: 14.94 MB | ESC-50 第 2-5 折,47 类/1504 条有效音频<br>标签投影宏平均 AP<br>ONNX: 74.15% | **W8A16**<br>DLA: 3.99 MB<br>NPU: 0.516 ms<br>RSS: 10.75 MiB | 73.90%<br>下降 0.25 个百分点<br>[报告](audio/ambient_sound/yamnet/results/summary.json) | — | — |
| YOLOv8n + OSNet_x0_25 + BoT-SORT（BoxMOT） | SLAM / 目标跟踪 | — | — | — | — | — | — | — |
| MobileNetVLAD（mobilenetvlad_depth-0.35） | SLAM / 定位 | — | — | — | — | — | — | — |
| HF-Net | SLAM / 定位 | — | — | — | — | — | — | — |
| SuperPoint（MagicLeap pretrained） | SLAM / 定位 | — | — | — | — | — | — | — |
| LightGlue（SuperPoint，固定1024关键点） | SLAM / 定位 | — | — | — | — | — | — | — |
| PointPillar-Tiny | 点云 / 点云 | — | — | — | — | — | — | — |
| CenterPoint-Pillar 0.2（双 PFN,10 sweeps）<br>[tianweiy/CenterPoint](https://github.com/tianweiy/CenterPoint) | 点云 / 点云 | 每点 5 维,10 sweeps<br>参数量待统计 | — | nuScenes,10 类<br>mAP / NDS | — | — | — | — |
| MIT BEVFusion | 点云 / 3D融合 | — | — | — | — | — | — | — |
| [YOLOv5s-v7.0](perception/object_detection/yolov5s/README.md)<br>YOLOv5-S,官方 `v7.0` 权重 | 感知 / 目标检测 | RGB `1×3×640×640`<br>7.24 M | 原始: 14.81 MB<br>ONNX: 28.94 MB | COCO val2017,5000 张<br>bbox mAP@0.5:0.95<br>ONNX: 37.09% | **W8A8**<br>DLA: 7.66 MB<br>NPU: 9.757 ms<br>RSS: 33.23 MiB | 35.86%<br>下降 1.23 个百分点<br>[报告](perception/object_detection/yolov5s/results/summary.json) | — | — |

### 数据说明

- 输入按 `N×C×H×W` 表示,图像 batch=1;音频采样率为 16 kHz.参数量 `1 M = 100 万`,不含 BatchNorm 运行统计量等缓冲区.
- 文件大小采用十进制 MB,`1 MB = 1,000,000 bytes`;RSS 为 MiB,与文件大小分开记录.原始权重为当前官方/原作者检查点文件实际大小,保留格式,不将其等同于 FP32 网络大小.多分片、双模型按实际必需文件合计;DLA 不含外置 embedding,有外置权重时另列部署合计.
- `WnAm` 表示权重 n 位、激活 m 位.浮点必须同时标明 FP16/FP32,不可与同位宽整数定点量化混淆;W8A8 为 INT8 PTQ,主要权重采用逐输出通道量化,不表示偏置、全部算子或文件 I/O 都是 8 位.本表不统计仅有浮点部署结果的版本.
- 精度仅比较同模型、同数据集与同评测协议的参考基线.表中直接写“上升/下降”,准确率/AP/AR 越高越好,WER/PPL 越低越好;PPL 差值不是百分点.差值按报告原始数值计算,不由展示后舍入值相减.MiniCPM5 使用原框架 FP32 参考,不是 ONNX 基线.
- NPU 耗时按一次模型推理统计.图像模型 batch=1;RTMPose 一次输入为一个裁剪好的人体,YAMNet 一次输入为 96 帧 Log-Mel 特征,不等同于整段音频处理耗时.
- MiniCPM 的 Prefill 是输入处理阶段,Decode 是逐 Token 生成阶段.表中速度来自 128 Token 输入、最多生成 128 Token 的测速协议;当前 RSS 来自该次生成性能进程,不是设备总内存.DeepSeek 的速度和 RSS 来自中英文生成样例,与 MiniCPM 的定长测速协议不同.
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
