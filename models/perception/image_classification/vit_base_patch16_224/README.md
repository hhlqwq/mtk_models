# ViT-Base Patch16 224

## 模型信息

```text
模型: ViT-Base Patch16 224
任务: ImageNet-1K 图像分类
输入: 1×3×224×224 RGB
输出: 1×1000 classes
设备: MediaTek Genio 720 EVK
当前状态: Genio 720 INT8 转换、部署和 ImageNet 全量评测已完成
```

> 迁移说明：正式上游已改为 PyTorch Vision v0.15.1 的
> `ViT_B_16_Weights.IMAGENET1K_V1`.本目录旧 Qualcomm v0.61.0 ONNX 产物和结果仅
> 保留为历史工程证据,不计入当前交付状态.

## 目标执行流程

```bash
cd /workspace/models/perception/image_classification/vit_base_patch16_224
# 使用本地官方 .pth 权重离线校验并自行导出 ONNX.
./deploy/download_original.sh
./deploy/convert.sh
./deploy/build.sh
./deploy/deploy_board.sh
```

以下为历史 Qualcomm ONNX 衍生模型的验证记录.正式 Top-1 使用 ImageNet ILSVRC2012
验证集及可靠标签映射.历史流程已完成 50,000 张
FP32 ONNX / MTK NPU INT8 绝对精度评测, 并单独报告排除 100 张 PTQ 校准图片后的
49,900 张独立指标. 完整结果和证据哈希见 `docs/accuracy.md` 与
`docs/imagenet_accuracy_20260908.json`.

官方标签映射使用以下两个小型资源, 不需要重新下载验证图片:

| 文件 | 官方地址 | SHA-256 |
| --- | --- | --- |
| `ILSVRC2012_devkit_t12.tar.gz` | `https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz` | `b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953` |
| `imagenet_class_index.json` | `https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json` | `a1e7a966a1f601d39e4b43e119b3e7dd4a2ad3ea08cf69847cbaf021013767bc` |

将文件放入 `original/imagenet_eval/`, 在 89 容器中执行:

```bash
python /workspace/tools/accuracy/imagenet_val_labels.py \
    --devkit original/imagenet_eval/ILSVRC2012_devkit_t12.tar.gz \
    --class-index original/imagenet_eval/imagenet_class_index.json \
    --qualcomm-labels original/labels.txt \
    --output original/imagenet_eval/imagenet_val_labels_0based.txt \
    --manifest original/imagenet_eval/imagenet_val_labels_manifest.json
```

生成器通过 devkit 的 `ILSVRC2012_ID -> WNID` 和 Keras/TensorFlow 的
`WNID -> 0-based 输出索引` 建立映射, 并校验 50,000 张、1000 类及每类 50 张.
已生成标签的 SHA-256 为
`098d797749a19d2c76f3243494b4d38079446eab41f3b8775212a33e4558a35f`,
Qualcomm 显示名称与输出索引映射的差异数为 0. 可公开复核摘要见
`docs/imagenet_label_mapping.json`. 受 ImageNet 条款约束的原始和派生标签文件
不进入普通 Git 历史.

## 交付状态

| 环节 | 状态 | 证据 |
| --- | --- | --- |
| 官方开源上游 | 已锁定 | PyTorch Vision v0.15.1、完整 SHA-256 已记录 |
| 原始框架基线 | 已锁定 | 官方发布 Top-1/Top-5 为 81.072%/95.318%,导出前完成数值一致性检查 |
| 自行导出 ONNX | 已完成 | 官方 `.pth` 经 TorchVision v0.15.1 离线加载与导出 |
| MTK 兼容 ONNX | 已完成 | 注意力已改写为 MTK 支持的固定四维计算并完成输出一致性检查 |
| MTK INT8 TFLite / DLA | 已完成 | 100 张独立 ImageNet 图片校准并通过 `mdla5.3` 编译 |
| 板端 Demo、精度和性能 | 已完成 | `20260910_vit_torchvision_v2`,ImageNet val 50,000 张 |

正式开源模型的 50,000 张结果为 FP32 ONNX Top-1/Top-5
80.64%/95.10%、Genio 720 NPU INT8 79.38%/94.69%.排除 100 张 PTQ 校准图片后,
49,900 张独立集的 NPU Top-1/Top-5 为 79.36%/94.68%.详细证据见
`docs/accuracy.md` 和 `docs/imagenet_accuracy_20260911.json`.

## 转换兼容性与评测约束

`download_original.sh` 只读取用户放置的官方 `.pth` 文件,校验完整 SHA-256 后通过
TorchVision v0.15.1 自行导出两个 opset 17 模型：精确 GELU 的 `model_fp32.onnx`
作为 FP32 基线,标准 tanh GELU 的 `model_mtk_compatible.onnx` 作为 MTK 转换候选.
脚本使用固定随机输入限制最大绝对 logit 偏差、平均绝对偏差和 Top-1 漂移；任何一项
超限都会停止.是否仍需其他 MTK 图改写,必须在 89 上实际导出后根据新图确认.

新导出 ONNX 在图内执行 TorchVision ImageNet mean/std 归一化,外部输入固定为 NCHW RGB
float32 `[0,1]`. `convert.sh` 默认使用排序后的 ImageNet val 第 1001~1100 张
校准, 与默认前 1000 张对齐评测子集错开. 已生成映射到模型输出顺序的 50,000 条
0-based 标签. 正式绝对精度报告已同时披露完整 50,000 张指标, 以及排除其中 100 张
PTQ 校准图片后的 49,900 张独立指标; 默认 1000 张结果保留为早期后端一致性证据.

评测采用独立运行目录 `.eval/vit_base_patch16_224/runs/<run_id>/`. `all` 自动
创建运行 ID; 分阶段执行必须为 `prepare`、`board`、`compare` 设置相同的
`EVAL_RUN_ID`. 模型、脚本、样本数或标签变化时必须使用新的运行 ID.
`START` 和 `TOTAL` 可选择连续子集; 完整 50,000 条标签可以直接用于任意子集,
评测器会严格校验图片文件名、官方图片编号、清单序号和标签序号的一致性.
提供标签时默认同时报告全部样本指标, 以及排除第 1001~1100 张 PTQ 校准图片的
独立指标. 排除区间可通过 `ACCURACY_EXCLUDE_START` 和
`ACCURACY_EXCLUDE_COUNT` 显式调整.

历史 Qualcomm v0.61.0 图的 IR/opset 降级、外部权重合并和 CUDA Provider 限制只保留
为旧工程证据,不适用于新的 PyTorch Vision 导出模型.新模型需要使用新的运行 ID 重新
完成 ONNX Runtime、MTK Converter、NCC 和板端验证.
