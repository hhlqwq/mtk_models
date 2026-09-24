# Depth Anything V2 Small 单目相对深度

本目录用于 Genio 720 的单图硬件冒烟。官方 Small 权重已完成 518×518 转换、MDLA 5.3 无桥接编译与三次板端硬件推理，当前状态为 `board_verified`。输出是**相对深度**，不能直接当作米制距离或安全避障阈值。

## 官方来源

- 源码：[DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)，固定提交 `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`。
- 权重：[Depth-Anything-V2-Small/depth_anything_v2_vits.pth](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/blob/03876f8651c73a60fe4c2c48294e09fcb6838fcf/depth_anything_v2_vits.pth)，99,218,434 字节，SHA-256 `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`。
- 许可：官方声明 Small 为 Apache-2.0。以上权重下载时官方域名连接超时，89 测试机经 `https://hf-mirror.com/depth-anything/Depth-Anything-V2-Small/resolve/03876f8651c73a60fe4c2c48294e09fcb6838fcf/depth_anything_v2_vits.pth` 获取；文件 SHA-256 与镜像响应的 `x-linked-etag` 一致。

## 冒烟配置

采用固定 518×518 RGB 输入，按 ImageNet 均值和标准差归一化，输出 518×518 相对深度。518 是官方默认推理边长；本次固定方形缩放用于验证转换和硬件推理，官方实现会保留宽高比，正式精度评估须重新确定协议。

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

运行编号 `20260924T092605Z`。两张图片的板端相对深度图分别与同输入 PyTorch 参考达到 0.995495、0.990525 的逐像素 Pearson 相关系数；重复输入的原始输出逐字节一致。板端绝对输出幅值与 PyTorch 有差别，正式精度、稳定延迟和 Genio 5100 仍待评估。具体命令参数、哈希和原始证据见 [冒烟记录](docs/smoke.md)。

![板端相对深度预览](examples/output/public/depth_anything_v2_small_sample_1.png)
