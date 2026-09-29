# 精度评估

DA-2K 全量点对精度已完成。运行编号 `20260928_depth_da2k_full_v1`，板端处理 1,033/1,033 张图、2,068 个点对，正确 1,774 对，准确率 `0.8578336557`。[完整报告](../results/full_accuracy/20260928_depth_da2k_full_v1/summary.json)。输入固定缩放到 518×518，输出双线性还原后读取标注点；该协议与官方保留宽高比的配置不同，也不测量米制深度误差。

数据源为[官方 DA-2K](https://huggingface.co/datasets/depth-anything/DA-2K/tree/main)，实际压缩包地址为 `https://hf-mirror.com/datasets/depth-anything/DA-2K/resolve/main/DA-2K.zip`。压缩包 SHA-256 `ff0e48e7cc53273efd1312610e51f1ec87bea0b8a22daf37125fd81246592b81`，与[官方文件页](https://huggingface.co/datasets/depth-anything/DA-2K/blob/main/DA-2K.zip)一致；标注 SHA-256 `830d51c1145198593a0f59e19b0f3c3b53cea725deaf6494e25f9b3ab35320cb`。板端路径 `/root/hailong.he/datasets/da2k/`。

先前 518×518 双图冒烟的 PyTorch/板端逐像素 Pearson 相关系数为 0.995495、0.990525，见 [smoke.md](smoke.md)。同协议 PyTorch、ONNX 全量精度对照及错误分布仍待补充，因此状态保持 `board_verified`。
