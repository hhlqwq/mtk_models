# YOLOv5s

## 模型信息

```text
模型: YOLOv5s
任务: COCO 80 类目标检测
输入: 1×3×640×640 RGB
输出: 3 个检测头
设备: MediaTek Genio 720 EVK
部署格式: INT8 TFLite → DLA
当前状态: 环境建设中
```

Qualcomm Hugging Face 页面用于对标交付形式；由于其页面当前描述的是 YOLOv5-M 且不分发
预导出资产，本模型采用 Ultralytics YOLOv5s 上游权重，并按照 MTK 官方 YOLOv5s 流程转换。

## 执行流程

```bash
cd /workspace/models/perception/object_detection/yolov5s
./deploy/download_original.sh
./deploy/convert.sh
./deploy/build.sh
./deploy/deploy_board.sh
```

每一步会检查上一步产物，失败后立即停止。`convert.sh` 需要
`/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017/images`
作为校准图片来源；正式精度使用完整 COCO val2017，不复用校准结果冒充 mAP。

## 交付状态

| 环节 | 状态 | 证据 |
| --- | --- | --- |
| 来源锁定 | 待执行 | `original/source_url.txt` |
| 原始模型 | 待执行 | `models/yolov5s.pt` |
| ONNX | 待执行 | `models/model_fp32.onnx` |
| INT8 TFLite | 待执行 | `models/model_int8.tflite` |
| DLA | 待执行 | `models/model_int8.dla` |
| 板端 Demo | 待执行 | `examples/output/` |
| 正式精度 | 待执行 | `docs/accuracy.md` |
| 正式性能 | 待执行 | `docs/benchmark.md` |
