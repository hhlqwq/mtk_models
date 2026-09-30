# Depth Anything V2 Small 精度

DA-2K 全量 1033 张图片、2068 个点对。主指标为点对准确率。

| 后端 | 点对准确率 |
| --- | ---: |
| FP32 ONNX | 94.83% |
| MTK NPU INT8 | 85.78% |

板端较 ONNX 低 9.04 个百分点，量化误差仍需分析。两端使用相同的 518×518 方形缩放、双线性还原及点对判定协议；该口径不同于官方保留宽高比的配置，也不表示米制深度误差。结果见[参考报告](../results/reference_accuracy/da2k_fp32_v1/summary.json)和[板端报告](../results/full_accuracy/20260928_depth_da2k_full_v1/summary.json)。数据来源为 [DA-2K](https://huggingface.co/datasets/depth-anything/DA-2K/tree/main)。
