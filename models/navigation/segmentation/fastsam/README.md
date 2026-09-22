# FastSAM-s / Genio 720

当前为 **环境建设中**: 已编写离线导出、INT8 量化、DLA 编译、板端 Demo 与同图比较脚本;
尚未取得官方权重,未完成模型导出、编译或板端推理,不能称为适配验证完成.

采用 FastSAM-s、batch=1、640×640.模型生成类别无关实例掩码,
支持全图输出与轻量点/框提示,暂不接入文本提示所需的 CLIP.
G5100 保持未开始状态.

## 官方资源准备

- 模型作者: [CASIA-LMC-Lab/FastSAM](https://github.com/CASIA-LMC-Lab/FastSAM).
- 权重: 官方 README 指向的 [FastSAM-s.pt](https://drive.google.com/file/d/10XmSj6mmpmRb8NhXbtiuO9cTTBwR_9SV/view?usp=sharing).
- 导出实现: 复用容器已安装的 [Ultralytics v8.0.111](https://github.com/ultralytics/ultralytics/tree/v8.0.111).
- 下载地址与来源版本见 [source_url.txt](original/source_url.txt).

用户在本机下载并放到:

```text
D:\code\github\mtk_models\models\navigation\segmentation\fastsam\original\FastSAM-s.pt
```

记录实际文件大小与 SHA-256,再同步到 89 的对应 `original/FastSAM-s.pt`.
`download_original.sh` 只校验离线文件,不会发起下载.
当前权重哈希为空; 到位后补齐来源记录和 model.yaml,不能将本地计算哈希冒充上游公布哈希.

## 转换与编译

以下命令在 89 的现有 `hhl_g720_8011` 容器执行.不升级共享容器依赖.
权重哈希填写在核验后的环境变量中,不能保留示例占位文字.

```bash
cd /data/users/hailong.he/github/mtk_models
export FASTSAM_WEIGHTS_SHA256='<核验后的64位SHA256>'
export FASTSAM_IMAGE='/data/users/hailong.he/github/mtk_models/models/perception/object_detection/yolov5s/examples/input/public/000000000001.jpg'
export FASTSAM_CALIBRATION_DIR='/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017/images'
bash models/navigation/segmentation/fastsam/deploy/download_original.sh
bash models/navigation/segmentation/fastsam/deploy/convert.sh
bash models/navigation/segmentation/fastsam/deploy/build.sh
```

导出 opset 13,10 个原始输出分别保留三个尺度的框 logits、分数 logits、
掩码系数及原型.DFL、NMS 和掩码还原在 CPU 完成,不编入 DLA.
原型分支包含转置卷积,其 MTK 转换与 MDLA 支持仍需真实编译核实.
编译参数采用 Genio 720 已有流程的 `--arch=mdla5.3 --suppress-output --disallow-bridge`.

## 板端验证

先核实 92 的 SSH 主机密钥和 Python 依赖.以下命令在 89 宿主执行,
`FASTSAM_IMAGE` 必须与导出基线使用的图片相同:

```bash
cd /data/users/hailong.he/github/mtk_models
export FASTSAM_IMAGE='/data/users/hailong.he/github/mtk_models/models/perception/object_detection/yolov5s/examples/input/public/000000000001.jpg'
export FASTSAM_RUN_ID="$(date +%Y%m%d_%H%M%S)"
bash models/navigation/segmentation/fastsam/deploy/deploy_board.sh
```

流程在 `/root/hailong.he/fastsam/runs/<run_id>` 执行 ONNX CPU 基线、PyTorch/ONNX
比较、NPU 推理、ONNX/NPU 比较与 100 次 runtime 性能采集.
输出保存在 `examples/output/runs/<run_id>`; 同名板端目录存在时拒绝覆盖.
NPU 量化一致性报告不会自动给出合格结论,需审核误差、漏检和可视化结果.

详细说明: [Demo](deploy/inference_demo/README.md)、[精度](docs/accuracy.md)、
[性能](docs/benchmark.md)、[模型卡](model_card.md).

## 当前阻塞

- 官方 FastSAM-s.pt 尚未下载,真实权重与 Ultralytics 8.0.111 的兼容性待确认.
- 89 到 92 的 SSH 主机密钥因刷机发生变化,待核实后记录.
- 新系统的板端 Python、NumPy、OpenCV、ONNX Runtime 依赖待重新核实.
- 正式数据集长评测不自动启动,与小样本一致性验证分开维护.

## 静态检查记录

2026-09-22: 6 个 Python 文件通过 AST 语法解析,4 个 Shell 脚本通过 `bash -n`.
FastSAM 注册条目和目录必需文件检查通过.全仓 `tools/check_registry.py` 仍因已有
Whisper 条目的 `board_validated` 不在允许状态列表中失败; 本次未修改其他模型状态.
这些检查不包含模型执行,不能作为转换、后处理数值或 NPU 运行验证.
