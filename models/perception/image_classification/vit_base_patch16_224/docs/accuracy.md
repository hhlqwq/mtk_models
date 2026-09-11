# ViT-Base 精度报告

## 正式开源模型结果

2026-09-11 完成 TorchVision v0.15.1 `ViT_B_16_Weights.IMAGENET1K_V1`
自产 ONNX 和 Genio 720 INT8 DLA 的 ILSVRC2012 val 50,000 张全量评测.运行 ID 为
`20260910_vit_torchvision_v2`,FP32 基线使用 ONNX Runtime CUDA Provider,板端使用
Neuron Runtime 8.2.16.

| 指标 | FP32 ONNX | MTK NPU INT8 | INT8 差值 |
| --- | ---: | ---: | ---: |
| Top-1 | 80.64% | 79.38% | -1.26 个百分点 |
| Top-5 | 95.10% | 94.69% | -0.41 个百分点 |
| 排除校准集 Top-1 | 80.63% | 79.36% | -1.27 个百分点 |
| 排除校准集 Top-5 | 95.10% | 94.68% | -0.42 个百分点 |

后端 Top-1 一致率为 91.68%,Top-5 集合平均重合度为 78.29%,logit 最大绝对误差
5.6219、RMSE 0.3386、相对 Frobenius 误差 0.5351.排除校准集指标使用 49,900
张图片,排除 `convert.sh` 使用的第 1001~1100 张图片.

完整性标记 `prepare.done`、`board.done` 和 `agreement_summary.json` 均存在,
Manifest 为 50,000/50,000.89 证据位于
`.eval/vit_base_patch16_224/runs/20260910_vit_torchvision_v2/`,板端原始输出位于
`/root/hailong.he/vit_eval/runs/20260910_vit_torchvision_v2/`.机器可读摘要见
`imagenet_accuracy_20260911.json`.

当前正式模型产物 SHA-256：FP32 ONNX
`5cf7159b106ef651b0800c0b75b04629ebf173c44e18b9a018a4e38d1d89ce7c`,MTK
兼容 ONNX `df15d9dad8e9952a865ff64e431e24baa4f64884d2006a1a0e4d7db3a0899692`,
INT8 TFLite `74976e225c06ffb1e82d878360e70ad267eba17b819a4e793ff5ced0afa330c1`,
DLA `2aec455949fa25420611645eb97617433046b6d488d87e66edf26e8246b85d67`.

> 历史结果说明：本页结果来自 Qualcomm v0.61.0 预导出 FP32 ONNX 及其 MTK INT8
> 衍生模型.它们保留用于工程对照,不属于待重新锁定的开源上游 ViT 正式交付结果.

历史状态：已完成 50,000 张 FP32 ONNX / MTK NPU INT8 绝对精度及后端对齐评测.

| 后端 | ImageNet Top-1 | ImageNet Top-5 | 验证集 |
| --- | ---: | ---: | --- |
| 历史来源 PyTorch | 不适用 | 不适用 | 历史交付源为 Qualcomm FP32 ONNX |
| FP32 ONNX | **80.64%** | **95.11%** | ILSVRC2012 val 50,000 |
| MTK NPU INT8 | **79.40%** | **94.64%** | ILSVRC2012 val 50,000 |

## 完整验证集绝对精度

2026-09-08 完成 ILSVRC2012 val 全部 50,000 张图片评测. FP32 基线为未经 GELU
近似的 Qualcomm 原始 ONNX, 显式使用 ONNX Runtime 1.18 CPU Provider; NPU 为
MT8189 / MDLA 5.3 INT8 DLA, 由板端 Neuron Runtime 8.2.16 执行.

| 指标 | FP32 ONNX | MTK NPU INT8 | INT8 差值 |
| --- | ---: | ---: | ---: |
| Top-1 | 80.64% | 79.40% | -1.24 个百分点 |
| Top-5 | 95.11% | 94.64% | -0.47 个百分点 |
| 排除校准集 Top-1 | 80.63% | 79.38% | -1.25 个百分点 |
| 排除校准集 Top-5 | 95.10% | 94.63% | -0.47 个百分点 |

排除校准集指标使用 49,900 张图片, 排除了 `convert.sh` 默认用于 PTQ 校准的
第 1001~1100 张图片. 完整集与独立集结果接近, 但完整 50,000 张仍是标准验证集
口径, 49,900 张结果用于披露校准数据重叠影响.

本次运行 ID 为 `20260908_imagenet50000_absolute_v1`. 完整性检查结果如下:

| 证据 | 结果 |
| --- | ---: |
| Manifest | 50,000/50,000 |
| NPU 输入 | 50,000/50,000 |
| FP32 logits | 50,000/50,000 |
| 回传的非空 NPU 输出 | 50,000/50,000 |
| 板端运行日志 | 50,000/50,000 完成 |
| Top-1 agreement | 91.68% |
| Top-5 集合平均重合度 | 78.23% |
| Logit 最大绝对误差 | 5.8413 |
| Logit RMSE | 0.3407 |
| Logit 相对 Frobenius 误差 | 0.5384 |

89 证据位于
`.eval/vit_base_patch16_224/runs/20260908_imagenet50000_absolute_v1/`, 92 原始 NPU
输出保留在 `/root/hailong.he/vit_eval/runs/20260908_imagenet50000_absolute_v1/`.
运行配置锁定模型、DLA、评测脚本、标签、样本数和 Provider 哈希. 可公开的机器可读
摘要见 `imagenet_accuracy_20260908.json`; `.eval`、数据集及标签正文不进入 Git.

## 后端一致性结果

2026-09-08 在 ImageNet val 排序后的前 1000 张图片上完成对齐评测. PTQ 校准使用
第 1001~1100 张, 与该对齐子集完全错开. FP32 基线为未经 GELU 近似的 Qualcomm
原始 ONNX, 显式使用 ONNX Runtime 1.18 CPU Provider; NPU 为 MT8189 / MDLA 5.3
INT8 DLA, 由板端 neuronrt 8.2.16 执行.

| 指标 | 结果 |
| --- | ---: |
| 完整样本 | 1000/1000 |
| Top-1 agreement | **92.3%** |
| Top-5 集合平均重合度 | 78.28% |
| Logit 最大绝对误差 | 4.1271 |
| Logit RMSE | 0.3422 |
| Logit 相对 Frobenius 误差 | 0.5428 |

运行 ID 为 `20260908_agreement1000`. 89 证据位于
`.eval/vit_base_patch16_224/runs/20260908_agreement1000/`, 92 原始 NPU 输出位于
`/root/hailong.he/vit_eval/runs/20260908_agreement1000/`. 运行目录绑定模型、脚本、
样本数和 Provider 哈希, 且逐项校验 1000 个 FP32 logits 与 NPU 输出均存在且大小正确.

该结果说明 INT8 PTQ 存在 7.7% 的 Top-1 预测漂移. Agreement 不能直接换算成
ImageNet Top-1/Top-5; 绝对精度必须使用逐图 ground truth 单独计算.

## 官方标签映射

2026-09-08 使用 ImageNet 官方 devkit ground truth、devkit `meta.mat` 和
Keras/TensorFlow ImageNet 类索引生成 50,000 条 0-based 输出标签. 映射链为
`ILSVRC2012_ID -> WNID -> model_output_index_0based`. 完整性结果如下:

| 项目 | 结果 |
| --- | ---: |
| 标签数量 | 50,000 |
| 类别范围 | 0–999 |
| 每类样本数 | 50 |
| Qualcomm 显示名称差异 | 0 |
| 标签文件 SHA-256 | `098d797749a19d2c76f3243494b4d38079446eab41f3b8775212a33e4558a35f` |

来源 URL、输入文件 SHA-256 和机器可读结果见 `imagenet_label_mapping.json`.
原始和派生标签文件不进入普通 Git 历史.
