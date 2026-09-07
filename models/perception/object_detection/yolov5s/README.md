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

89 服务器和 92 开发板不得联网下载上述模型、数据集或源码文件。用户完成 GitHub 同步并
手动放置权重后，才能继续下面的流程。Docker 镜像和 Python 系统依赖仍可在 89 的 Docker
环境内按项目约定安装。

## 服务器执行流程

89 宿主机只在仓库目录执行命令：

```bash
cd /data/users/hailong.he/github/mtk_models
docker exec -it hhl_g720_311 bash
```

宿主机仓库以可写方式映射到 Docker 内的 `/workspace`。进入容器后执行：

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

`deploy/constraints-py311.txt` 固定 Python 3.11 转换依赖，防止上游宽松版本范围将 NumPy
升级到 MTK Converter 不支持的 2.x，或将 Ultralytics 升级到移除 `ultralytics.yolo`
命名空间的新版本。转换开始前会执行 `pip check` 和关键模块导入检查。

PyTorch 2.0.0 CUDA 11.8 使用 89 的 NVIDIA GPU 完成 TorchScript/ONNX 导出，PyTorch 和
ONNX 基线精度评测也使用 GPU。MTK Converter 8.16.0 的公开接口没有 CUDA/GPU 选项，
因此 INT8 PTQ 和 NCC 编译仍由 CPU 执行，不能将这两步描述为 GPU 加速。

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

## 全流程验收边界

YOLOv5s 只有同时完成以下项目才视为交付完成：

1. 在 `hhl_g720_311` 中生成 TorchScript、FP32 ONNX、INT8 TFLite 和 DLA。
2. 在 92 的 Genio 720 EVK 上加载 DLA 并完成真实图片推理和检测框后处理。
3. 报告预处理、纯 NPU、后处理及端到端延迟，并记录峰值内存。
4. 使用同一 COCO val2017 评测集分别测量 PyTorch、ONNX 和 MTK NPU 的 mAP。
5. 计算 ONNX 相对 PyTorch、MTK NPU INT8 相对 ONNX/PyTorch 的精度损失。

少量样例只能证明部署链路和输出合理性，不能替代完整 COCO val2017 精度报告。

板端 Demo 使用当前 TFLite 的实际量化参数生成输入，调用 92 的 `neuronrt 8.2.16` 完成
推理并回传三个检测头。性能阶段执行 10 次预热和 100 次连续推理，保存 Runtime 日志和
进程峰值 RSS；89 上再执行解量化、YOLOv5 解码、NMS 和结果绘制。
