# Depth Anything V2 Small 单目相对深度

## 模型信息

```text
模型: Depth Anything V2 Small
任务: 单目相对深度
输入: 1×3×518×518 RGB
输出: 518×518 相对深度图
设备: MediaTek Genio 720 EVK
部署格式: INT8 TFLite → DLA
当前状态: 板端已验证；稳定性能与应用层精度待补
```

本目录用于 Genio 720 的单目相对深度验证。官方 Small 权重已完成 518×518 转换、MDLA 5.3 无桥接编译、板端硬件冒烟和 DA-2K 全量点对评测。输出是**相对深度**，不能直接当作米制距离或安全避障阈值。

## 交付状态

| 环节 | 状态 | 证据 |
| --- | --- | --- |
| 官方权重与自行导出 | 已完成 | 固定版本与权重哈希见[来源](#官方来源) |
| MTK INT8 TFLite / DLA | 已完成 | 518×518 转换、MDLA 5.3 无桥接编译 |
| 板端 Demo | 已完成 | 两张图片与 PyTorch 同图对照见[冒烟报告](docs/smoke.md) |
| DA-2K 全量点对精度 | 已完成 | [参考端与板端结果](docs/accuracy.md) |
| 常驻实例性能 | 待补 | 当前仅有逐图 CLI 墙钟耗时，见[性能报告](docs/benchmark.md) |

## DA-2K 正式精度入口

本目录新增 `deploy/run_full_accuracy.sh`，在 Ubuntu89 宿主机以
`EVAL_RUN_ID=<新ID> bash deploy/run_full_accuracy.sh` 启动。它调用 92 上的硬件 DLA，
按官方 DA-2K 压缩包的全部 1,033 张标注图片、2,068 个点对计算相对深度排序准确率。
评测输入复用冒烟时验证的 518×518 正方形缩放和 INT8 行补齐；输出双线性还原到原图尺寸，
再读取官方标注点坐标。点对中 `point1` 的深度数值严格大于 `point2` 时计为正确。
此协议是本项目固定输入形状的板端评测，不能直接与官方保留宽高比的推理配置横比。
报告保留在板端 `open_models/depth_anything_v2_small/eval/<新ID>/report/`，
中断后以同一 ID 加 `EVAL_RESUME=1` 续跑；脚本不删除数据、原始输出或报告。

`20260928_depth_da2k_full_v1` 已完成 1,033 张图、2,068 个点对，正确 1,774 对，
点对准确率 85.78%；每图 `neuronrt` 命令平均耗时 194.70 ms。
详细数据与哈希见[板端报告](results/full_accuracy/20260928_depth_da2k_full_v1/summary.json)。
同协议 FP32 ONNX 全量参考准确率为 94.83%，板端低 9.04 个百分点；详见[精度报告](docs/accuracy.md)。

数据来源为 [官方 DA-2K 数据集](https://huggingface.co/datasets/depth-anything/DA-2K/tree/main)，
实际获取地址为
`https://hf-mirror.com/datasets/depth-anything/DA-2K/resolve/main/DA-2K.zip`；
压缩包 SHA-256 为 `ff0e48e7cc53273efd1312610e51f1ec87bea0b8a22daf37125fd81246592b81`，
与[官方文件页](https://huggingface.co/datasets/depth-anything/DA-2K/blob/main/DA-2K.zip)一致。
数据集许可见官方页面的 Apache-2.0 声明。板端数据路径固定为
`/root/hailong.he/datasets/da2k/`。运行前需准备完整数据，入口不会下载或解压。
报告中的 `cli_wall_ms` 包含每张图片启动 `neuronrt` 与模型加载，不表示纯 NPU 延迟。

## 官方来源

- 源码：[DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)，固定提交 `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`。
- 权重：[Depth-Anything-V2-Small/depth_anything_v2_vits.pth](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/blob/03876f8651c73a60fe4c2c48294e09fcb6838fcf/depth_anything_v2_vits.pth)，99,218,434 字节，SHA-256 `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`。
- 许可：官方声明 Small 为 Apache-2.0。以上权重下载时官方域名连接超时，89 测试机经 `https://hf-mirror.com/depth-anything/Depth-Anything-V2-Small/resolve/03876f8651c73a60fe4c2c48294e09fcb6838fcf/depth_anything_v2_vits.pth` 获取；文件 SHA-256 与镜像响应的 `x-linked-etag` 一致。

## 冒烟配置

采用固定 518×518 RGB 输入，按 ImageNet 均值和标准差归一化，输出 518×518 相对深度。518 是官方默认推理边长；本次固定方形缩放用于转换、硬件推理及 DA-2K 评测，官方实现会保留宽高比，两种协议的精度不能直接横比。

转换在 Ubuntu 89 的 `hhl_g720_8011` 容器执行，板端推理在 Genio 720 `192.168.0.92` 执行。原始权重、中间模型及运行输出保留在测试机，不提交 Git。源码位置编码在固定形状下预计算，注意力改写为四维张量，复用官方参数；导出脚本先检查改写前后的 PyTorch 输出。`--suppress-input` 的 INT8 输入每行须从 518 补齐到 528 字节，准备脚本负责补齐。

## 复现流程

在 89 宿主机执行资源准备；若文件已存在，脚本只核验版本和哈希：

```bash
cd /data/users/hailong.he/github/mtk_models/models/navigation/single_camera_depth/depth_anything_v2_small
bash deploy/fetch_original.sh
```

在现有 `hhl_g720_8011` 容器中执行转换和编译：

```bash
cd /data/users/hailong.he/github/mtk_models/models/navigation/single_camera_depth/depth_anything_v2_small
bash deploy/convert.sh
bash deploy/build.sh
```

回到 89 宿主机执行板端冒烟；需要 89 到 92 的免密 SSH：

```bash
cd /data/users/hailong.he/github/mtk_models/models/navigation/single_camera_depth/depth_anything_v2_small
bash deploy/deploy_board.sh
```

脚本复用 YOLOv5s 两张公共 CC0 图片，第三次重复第一张。新运行编号的原始证据保存在 `examples/output/runs/<运行编号>/`。重新转换会产生新的模型哈希，不应与本次记录混用。

## 本次结果

运行编号 `20260924T092605Z`。两张图片的板端相对深度图分别与同输入 PyTorch 参考达到 0.995495、0.990525 的逐像素 Pearson 相关系数；重复输入的原始输出逐字节一致。板端绝对输出幅值与 PyTorch 有差别；全量精度见[精度报告](docs/accuracy.md)，稳定延迟和 Genio 5100 仍待评估。冒烟证据见 [smoke.md](docs/smoke.md)，全量耗时见[性能报告](docs/benchmark.md)。

![板端相对深度预览](examples/output/public/depth_anything_v2_small_sample_1.png)
