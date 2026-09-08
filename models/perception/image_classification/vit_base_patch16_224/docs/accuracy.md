# ViT-Base 精度报告

状态：已完成 1000 张 FP32 ONNX / MTK NPU INT8 对齐评测; 绝对精度等待可靠标签映射.

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

该结果说明 INT8 PTQ 存在 7.7% 的 Top-1 预测漂移. 在缺少可靠逐图 ground truth 时,
不能将 agreement 换算成 ImageNet Top-1/Top-5, 也不能标记为完整交付. Qualcomm 归档内
`labels.txt` 只是 1000 个输出类别名称, 不能替代 50,000 张验证图片的真实标签.
