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

## 本机离线输入

YOLOv5s 权重属于模型文件，必须由用户手动下载到本机工作区。正确的 Ultralytics v7.0
Release 地址为：

<https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt>

本机目标位置：

```text
D:\code\github\mtk_models\models\perception\object_detection\yolov5s\models\yolov5s.pt
```

不要使用已失效的 `ultralytics/assets/releases/download/v7.0/yolov5s.pt`。固定提交
`485da42273839d20ea6bdaf142fd02c1027aba61` 的 YOLOv5 源码包和 MTK 官方补丁包体积较小，
已在本机准备并纳入 Git：

| 文件 | SHA-256 |
| --- | --- |
| `original/yolov5-485da42.zip` | `ba30792a44660ae95bcc1e2fee1ca369b98871893894fc68ac2b69e26cdc7dae` |
| `original/model_conversion_YOLOv5s_example_20240916.zip` | `8cb3ee3f7059a522e47fecd1be6983404e455e3f7069c19e9fe0a750df6a6fb7` |
| `models/yolov5s.pt` | `8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee` |

89 服务器和 92 开发板不得联网下载上述文件。用户完成 GitHub 同步并手动放置权重后，才能
继续下面的流程。

## 服务器执行流程

```bash
cd /workspace/models/perception/object_detection/yolov5s
bash ./deploy/download_original.sh
bash ./deploy/convert.sh
bash ./deploy/build.sh
bash ./deploy/deploy_board.sh
```

`download_original.sh` 名称为兼容统一交付结构而保留，实际只执行 SHA-256 校验、离线解压和
MTK 补丁应用，不包含任何网络下载。`convert.sh` 按 MTK 官方 NeuroPilot Converter 流程，
先导出 TorchScript 和 FP32 ONNX，再从 TorchScript 执行 INT8 PTQ。

每一步会检查上一步产物，失败后立即停止。`convert.sh` 需要
`/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017/images`
作为校准图片来源；正式精度使用完整 COCO val2017，不复用校准结果冒充 mAP。

## 交付状态

| 环节 | 状态 | 证据 |
| --- | --- | --- |
| 来源锁定 | 已完成 | `original/source_url.txt`、两个固定版本压缩包及 SHA-256 |
| 原始模型 | 本机已准备，待用户同步 | `models/yolov5s.pt`，14,808,437 bytes，SHA-256 已记录 |
| ONNX | 待执行 | `models/model_fp32.onnx` |
| INT8 TFLite | 待执行 | `models/model_int8.tflite` |
| DLA | 待执行 | `models/model_int8.dla` |
| 板端 Demo | 待执行 | `examples/output/` |
| 正式精度 | 待执行 | `docs/accuracy.md` |
| 正式性能 | 待执行 | `docs/benchmark.md` |
