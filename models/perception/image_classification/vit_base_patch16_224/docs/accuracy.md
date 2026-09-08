# ViT-Base 精度报告

状态：已完成 1000 张 FP32 ONNX / MTK NPU INT8 对齐评测及 50,000 张官方标签映射;
绝对精度尚未执行.

| 后端 | ImageNet Top-1 | ImageNet Top-5 | 验证集 |
| --- | ---: | ---: | --- |
| 来源 PyTorch | 待测 | 待测 | ILSVRC2012 val 50,000 |
| FP32 ONNX | 待测 | 待测 | ILSVRC2012 val 50,000 |
| MTK NPU INT8 | 待测 | 待测 | ILSVRC2012 val 50,000 |

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
原始和派生标签文件不进入普通 Git 历史. 当前仅完成标签映射, 尚未执行 50,000 张
FP32 ONNX / MTK NPU INT8 绝对精度评测, 因此表中的 Top-1/Top-5 继续保持待测.
