# RTMPose Body2d

## 模型信息

```text
模型: RTMPose-Body2d
任务: WholeBody 人体姿态估计
输入: 1×3×256×192 RGB
输出: 133 个关节点的位置和置信度
设备: MediaTek Genio 720 EVK
当前状态: 环境建设中
```

RTMPose 是 top-down 姿态模型，只处理已经裁剪的人体框。完整图片 Demo 需要额外的人体检测器；
模型自身延迟和“检测器 + RTMPose”端到端延迟必须分别报告。

## 执行流程

```bash
cd /workspace/models/interaction/pose_detection/rtmpose_body2d
./deploy/download_original.sh
./deploy/convert.sh
./deploy/build.sh
./deploy/deploy_board.sh
```

## 交付状态

| 环节 | 状态 | 证据 |
| --- | --- | --- |
| Hugging Face 来源 | 已锁定 | `original/source_url.txt` |
| Qualcomm FP32 ONNX | 待下载 | `models/model_fp32.onnx` |
| MTK INT8 TFLite | 待执行 | `models/model_int8.tflite` |
| DLA | 待执行 | `models/model_int8.dla` |
| 板端 Demo | 待执行 | `examples/output/` |
| WholeBody AP | 等待确认数据集 | `docs/accuracy.md` |
| 板端性能 | 待执行 | `docs/benchmark.md` |
