# MobileFaceNet 人脸特征提取

本目录面向 Genio 720 上的机器人熟人识别。模型接收已检测、已对齐的人脸，输出特征向量；
检测、对齐、人员登记、阈值设定和多帧身份确认均属于应用流水线的其他环节。

## 来源和边界

- 模型设计：[MobileFaceNets 论文](https://arxiv.org/abs/1804.07573)。
- 固定开源实现：[`foamliu/MobileFaceNet` a687c71](https://github.com/foamliu/MobileFaceNet/tree/a687c71bea830e70d05fb3b38ddc7c68e1687e94)。这是第三方 PyTorch 实现，并非论文作者代码。
- 原始权重：[v1.0 `mobilefacenet.pt`](https://github.com/foamliu/MobileFaceNet/releases/download/v1.0/mobilefacenet.pt)，4,135,271 字节，SHA-256 `90a00ba1d8b0b688af3deb731ed53dca582e6106805d1bc3cfdef55f570493f4`。
- 仓库代码为 Apache-2.0，保留上游 `LICENSE`。权重使用范围和 MS-Celeb-1M 训练数据的产品使用条件仍需单独核实；当前部署只用于板端技术冒烟。
- `original/upstream/` 保存上述固定提交的原样 `mobilefacenet.py` 和 `config.py`。上游 v1.0 标签源码采用不同的 512 维结构，无法严格加载其 v1.0 发布权重；固定提交中的 128 维结构与权重参数布局对应。

## 输入输出

上游训练代码从 OpenCV 读取 **BGR** 图片，将 112×112 图像按 `(像素值 - 127.5) / 128` 转为 NCHW FP32。
输入必须是预先对齐的人脸，不能把整幅相机画面直接输入。本模型目标输出 128 维，
实际输出维度以固定权重的 PyTorch 和 ONNX 检查为准。

## 转换和板端冒烟

在 Ubuntu 89 的 `hhl_g720_8011` 容器中，从本目录执行：

```bash
bash deploy/convert.sh
bash deploy/build.sh
python deploy/prepare_input.py \
  --tflite models/model_int8.tflite \
  --image-dir examples/input/calibration \
  --output-dir examples/input/generated
bash deploy/deploy_board.sh
```

`examples/input/calibration` 需放至少 8 张上游仓库的 `*_aligned.jpg` 示例，仅在本地和测试机使用，
不提交人脸图片。`deploy/deploy_board.sh` 会重新生成量化输入，传送 DLA 到独立的板端运行目录，
执行三次真实 NPU 推理，其中第三次重复第一张输入。脚本回收输出并运行 `verify_board.py`，
验证输出长度、非零有限值与重复输入一致性；证据保存在 `examples/output/runs/<时间戳>/`。

Git 只管理部署脚本、上游源码快照、来源与使用说明。原始 `.pt` 权重、导出的 ONNX、TFLite、
DLA、人脸校准图片和板端原始输出均由 `.gitignore` 排除，留在本地、Ubuntu 89 和 Genio 720 板端。

冒烟通过只表示特征提取模型能在板端推理，不代表已完成人脸身份识别的精度、性能或产品验收。
