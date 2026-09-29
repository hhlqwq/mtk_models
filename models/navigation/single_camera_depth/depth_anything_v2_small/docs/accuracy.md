# 精度评估

DA-2K 全量点对精度已完成。板端运行 `20260928_depth_da2k_full_v1` 处理 1,033/1,033 张图、2,068 个点对，正确 1,774 对，准确率 `0.8578336557`。[板端报告](../results/full_accuracy/20260928_depth_da2k_full_v1/summary.json)。同数据、同 518×518 正方形缩放、双线性还原及点对判定规则下，FP32 ONNX CUDA 优先参考端正确 1,961/2,068 对，准确率 `0.9482591876`，[参考报告](../results/reference_accuracy/da2k_fp32_v1/summary.json)。板端低 `9.0426` 个百分点；这是真实的精度损失，不能因板端评测已完成而标为精度验收通过。此协议与官方保留宽高比的配置不同，也不测量米制深度误差。

数据源为[官方 DA-2K](https://huggingface.co/datasets/depth-anything/DA-2K/tree/main)，实际压缩包地址为 `https://hf-mirror.com/datasets/depth-anything/DA-2K/resolve/main/DA-2K.zip`。压缩包 SHA-256 `ff0e48e7cc53273efd1312610e51f1ec87bea0b8a22daf37125fd81246592b81`，与[官方文件页](https://huggingface.co/datasets/depth-anything/DA-2K/blob/main/DA-2K.zip)一致；标注 SHA-256 `830d51c1145198593a0f59e19b0f3c3b53cea725deaf6494e25f9b3ab35320cb`。板端路径 `/root/hailong.he/datasets/da2k/`。

先前 518×518 双图冒烟的 PyTorch/板端逐像素 Pearson 相关系数为 0.995495、0.990525，见 [smoke.md](smoke.md)。参考端使用本模型锁定的 FP32 ONNX，并逐图保存正确点对数量；量化损失成因与错误分布仍待分析，因此状态保持 `board_verified`。
