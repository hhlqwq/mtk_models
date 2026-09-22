# 模型目录

模型使用"Scenario → Category → Model"三级目录,与首批 Target AI Models 清单一致：

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

未开始的模型只保留在 `registry/target_models.yaml`,不预先创建大量空目录.模型进入实现阶段时,
使用 `tools/create_model.py` 创建完整交付目录,并加入 `registry/models.yaml`.

当前先行实现：

- `perception/object_detection/yolov5s`
- `perception/image_classification/vit_base_patch16_224`
- `interaction/pose_detection/rtmpose_body2d`

当前 MediaTek ONNX Runtime Model Zoo 接入：

- `perception/object_detection/yoloworld_xl`

当前适配中的分割模型：

- [`navigation/segmentation/fastsam`](navigation/segmentation/fastsam/README.md):
  FastSAM-s / 640×640,已提供离线转换与板端脚本,待官方权重和真实 NPU 验证.
