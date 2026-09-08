# ViT-Base Patch16 224

## 模型信息

```text
模型: ViT-Base Patch16 224
任务: ImageNet-1K 图像分类
输入: 1×3×224×224 RGB
输出: 1×1000 classes
设备: MediaTek Genio 720 EVK
当前状态: 环境建设中
```

## 执行流程

```bash
cd /workspace/models/perception/image_classification/vit_base_patch16_224
./deploy/download_original.sh
./deploy/convert.sh
./deploy/build.sh
./deploy/deploy_board.sh
```

正式 Top-1 需要 ImageNet ILSVRC2012 验证集及标签映射。数据集缺失时只允许执行模型转换和
单图 Demo，不填写最终 Top-1。

## 交付状态

| 环节 | 状态 | 证据 |
| --- | --- | --- |
| Hugging Face 来源 | 已锁定 | `original/source_url.txt` |
| Qualcomm FP32 ONNX | 待下载 | `models/model_fp32.onnx` |
| MTK INT8 TFLite | 待执行 | `models/model_int8.tflite` |
| DLA | 待执行 | `models/model_int8.dla` |
| 板端 Demo | 待执行 | `examples/output/` |
| ImageNet Top-1 | 等待数据集 | `docs/accuracy.md` |
| 板端性能 | 待执行 | `docs/benchmark.md` |

## 转换兼容性与评测约束

Qualcomm v0.61.0 归档中的 `vit.onnx` 使用外部权重文件, IR v10 / opset 21,
并包含 `Gelu` 和 `Reshape allowzero=1`. `download_original.sh` 会先校验归档
SHA-256, 合并外部权重, 再通过 `downgrade_onnx.py` 做 MTK Converter 可接受的
等价展开与兼容降级. 遇到动态 shape 或包含 0 的 `allowzero=1` 时脚本会拒绝
转换, 不进行猜测式改写.

Qualcomm ONNX 已在图内执行 mean/std 归一化, 外部输入固定为 NCHW RGB
float32 `[0,1]`. `convert.sh` 默认使用排序后的 ImageNet val 第 1001~1100 张
校准, 与默认前 1000 张对齐评测子集错开. 当前只有验证集, 因此正式 50,000 张
绝对 Top-1 评测需要另行提供已映射到模型输出顺序的 0-based 标签, 且需明确披露
其中 100 张曾参与 PTQ 校准; 默认 1000 张结果仅用于 FP32/INT8 后端一致性分析.

评测采用独立运行目录 `.eval/vit_base_patch16_224/runs/<run_id>/`. `all` 自动
创建运行 ID; 分阶段执行必须为 `prepare`、`board`、`compare` 设置相同的
`EVAL_RUN_ID`. 模型、脚本、样本数或标签变化时必须使用新的运行 ID.
