# ViT-Base Patch16 224

## 模型信息

```text
模型: ViT-Base Patch16 224
任务: ImageNet-1K 图像分类
输入: 1×3×224×224 RGB
输出: 1×1000 classes
设备: MediaTek Genio 720 EVK
当前状态: 已完成
```

## 执行流程

```bash
cd /workspace/models/perception/image_classification/vit_base_patch16_224
./deploy/download_original.sh
./deploy/convert.sh
./deploy/build.sh
./deploy/deploy_board.sh
```

正式 Top-1 使用 ImageNet ILSVRC2012 验证集及可靠标签映射. 当前已完成 50,000 张
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
| Hugging Face 来源 | 已锁定 | `original/source_url.txt` |
| Qualcomm FP32 ONNX | 已完成 | `models/model_fp32.onnx`, SHA-256 已记录 |
| MTK 兼容 ONNX | 已完成 | `models/model_mtk_compatible.onnx`, GELU tanh 近似偏差已校验 |
| MTK INT8 TFLite | 已完成 | `models/model_int8.tflite` |
| DLA | 已完成 | `models/model_int8.dla`, MDLA 5.3 |
| 板端 Demo | 已完成 | `examples/output/top5.json`, Top-1 `sea snake` |
| 1000 张对齐评测 | 已完成 | Top-1 agreement 92.3%, `docs/accuracy.md` |
| ImageNet 标签映射 | 已完成 | 50,000 张、1000 类、每类 50 张, `docs/imagenet_label_mapping.json` |
| ImageNet 绝对精度 | 已完成 | FP32 Top-1 80.64%, NPU Top-1 79.40%, `docs/accuracy.md` |
| 板端性能 | 已完成 | 纯 NPU 53.3903 ms/inf, `docs/benchmark.md` |

## 转换兼容性与评测约束

Qualcomm v0.61.0 归档中的 `vit.onnx` 使用外部权重文件, IR v10 / opset 21,
并包含 `Gelu` 和 `Reshape allowzero=1`. `download_original.sh` 会先校验归档
SHA-256 并合并外部权重为原始 FP32 基线 `model_fp32.onnx`. 另生成
`model_mtk_compatible.onnx`: 通过 `downgrade_onnx.py` 降级 IR/opset、清理安全的
`allowzero` 属性, 并将 MTK TFLite 导出器不支持的精确 Gelu 改为标准 tanh 近似.
脚本会用固定输入限制近似误差与 Top-1 漂移; 遇到动态 shape、包含 0 的
`allowzero=1`、最大绝对 logit 偏差超过 0.025、平均绝对偏差超过 0.005 或
Top-1 漂移时立即停止. 精度基线始终使用未近似的原始 FP32 ONNX.

Qualcomm ONNX 已在图内执行 mean/std 归一化, 外部输入固定为 NCHW RGB
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

原始 FP32 ONNX 的 opset 21 `Squeeze` 在当前 ONNX Runtime 1.18 CUDA Provider
中没有匹配内核, 因此评测脚本默认显式使用 CPU Provider, 不允许静默回退.
如后续环境已验证 CUDA 支持, 可设置 `FP32_PROVIDER=cuda` 并使用新的运行 ID.
