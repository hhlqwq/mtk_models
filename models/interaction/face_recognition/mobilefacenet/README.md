# MobileFaceNet 人脸特征提取

## 模型信息

```text
模型: MobileFaceNet
任务: 人脸特征提取与验证
输入: 1×3×112×112 RGB
输出: 人脸特征向量
设备: MediaTek Genio 720 EVK
部署格式: INT8 TFLite → DLA
当前状态: 板端已验证；常驻性能已测，应用层识别待补
```

本目录面向 Genio 720 上的机器人熟人识别。模型接收已检测、已对齐的人脸，输出特征向量；
检测、对齐、人员登记、阈值设定和多帧身份确认均属于应用流水线的其他环节。
模型已列入 [`registry/models.yaml`](../../../../registry/models.yaml)，Genio 720 状态为 `board_verified`。

## 交付状态

| 环节 | 状态 | 证据 |
| --- | --- | --- |
| 固定第三方实现与权重 | 已锁定 | 源码提交和权重哈希见[来源和边界](#来源和边界) |
| 自行导出 ONNX 与 INT8 DLA | 已完成 | 转换及板端冒烟见[流程](#转换和板端冒烟) |
| LFW 全量验证 | 重测中 | 旧结果使用错误的颜色及归一化协议，见[精度报告](docs/accuracy.md) |
| 常驻实例性能 | 已完成 | 预热后 100 次 Neuron Runtime 调用及峰值 RSS，见[性能报告](docs/benchmark.md) |
| 应用层识别 | 待补 | 检测、对齐、现场识别与多帧决策尚未评估 |

## LFW 正式验证入口

本目录新增 `deploy/run_full_accuracy.sh`。在 Ubuntu89 宿主机设置新
`EVAL_RUN_ID` 后运行该脚本，92 板端对 LFW 完整 10 折、6,000 对人脸执行硬件特征提取，
用其余 9 折选择余弦阈值，再计算留出折的验证准确率，同时记录逐图 CLI 耗时。
中断后使用相同 ID 加 `EVAL_RESUME=1` 续跑；原始输出、逐图检查点与报告保留在
`/root/hailong.he/open_models/mobilefacenet/eval/<新ID>/`。

`20260928_mobilefacenet_lfw_full_v1` 曾处理全部 6,000 对，但使用了与锁定上游
验证代码不一致的 BGR 和 `(x - 127.5) / 128` 输入。旧结果及哈希留在
[板端报告](results/full_accuracy/20260928_mobilefacenet_lfw_full_v1/summary.json) 供排查，
不作为正式精度。正确协议的参考端和板端结果正在重测，见[精度报告](docs/accuracy.md)。

数据使用 [LFW 图像与官方验证对的 Hugging Face 整理版](https://huggingface.co/datasets/marcelohaps/lfw)，
图像来自 `original_non_aligned` 变体，13,233 张。实际 Parquet 获取地址为
`https://hf-mirror.com/api/datasets/marcelohaps/lfw/parquet/default/train/0.parquet`，
SHA-256 为 `85ff8ac9530a935d2dc6f9e2933cfd72c79089b2f1c403c996b808ba5c07abcf`；
验证对来自该数据集提交 `12a61458b56d0433d07269dc1d64368abf4f6b4d` 的 `pairs.csv`，
SHA-256 为 `7f540157be42f57ab5bb1d7ef53b7b379e0331b4842cbd32f4d1b625243e54fe`。
`deploy/prepare_lfw_dataset.py` 使用 89 端 `pyarrow==17.0.0`，仅在临时目录还原图片并校验 13,233 图、
6,000 对和十折划分；板端数据放在 `/root/hailong.he/datasets/lfw/`，原始人脸图片不提交 Git。

**协议限制：**当前使用整理版的原始非对齐 250×250 图片，直接缩放至 112×112，
没有运行人脸关键点检测和对齐。因此本次结果只能作为该固定输入协议的板端验证，
不得与原作者或论文使用对齐人脸的 LFW 数值直接比较。`cli_wall_ms` 包含启动
`neuronrt` 与模型加载，并非纯 NPU 延迟。LFW 原始图片的使用条款需按其来源核对。

## 来源和边界

- 模型设计：[MobileFaceNets 论文](https://arxiv.org/abs/1804.07573)。
- 固定开源实现：[`foamliu/MobileFaceNet` a687c71](https://github.com/foamliu/MobileFaceNet/tree/a687c71bea830e70d05fb3b38ddc7c68e1687e94)。这是第三方 PyTorch 实现，并非论文作者代码。
- 原始权重：[v1.0 `mobilefacenet.pt`](https://github.com/foamliu/MobileFaceNet/releases/download/v1.0/mobilefacenet.pt)，4,135,271 字节，SHA-256 `90a00ba1d8b0b688af3deb731ed53dca582e6106805d1bc3cfdef55f570493f4`。
- 仓库代码为 Apache-2.0，保留上游 `LICENSE`。权重使用范围和 MS-Celeb-1M 训练数据的产品使用条件仍需单独核实；当前部署只用于板端技术冒烟。
- `original/upstream/` 保存上述固定提交的原样 `mobilefacenet.py` 和 `config.py`。上游 v1.0 标签源码采用不同的 512 维结构，无法严格加载其 v1.0 发布权重；固定提交中的 128 维结构与权重参数布局对应。

## 输入输出

锁定上游的 LFW 验证代码将 OpenCV 的 BGR 对齐结果转为 **RGB**，再按 ImageNet 均值
`[0.485, 0.456, 0.406]` 和标准差 `[0.229, 0.224, 0.225]` 归一化为 NCHW FP32。
输入必须是预先对齐的人脸，不能把整幅相机画面直接输入。本模型目标输出 128 维，
实际输出维度以固定权重的 PyTorch 和 ONNX 检查为准。

## 转换和板端冒烟

在 Ubuntu 89 的 `hhl_g720_8011` 容器中，从本目录执行转换和编译：

```bash
bash deploy/convert.sh
bash deploy/build.sh
```

在 Ubuntu 89 主机的同一目录执行板端冒烟，脚本会通过容器准备量化输入：

```bash
bash deploy/deploy_board.sh
```

`examples/input/calibration` 需放至少 8 张上游仓库的 `*_aligned.jpg` 示例，仅在本地和测试机使用，
不提交人脸图片。`deploy/deploy_board.sh` 会重新生成量化输入，传送 DLA 到独立的板端运行目录，
执行三次真实 NPU 推理，其中第三次重复第一张输入。脚本回收输出并运行 `verify_board.py`，
验证输出长度、非零有限值与重复输入一致性；证据保存在 `examples/output/runs/<时间戳>/`。

Git 只管理部署脚本、上游源码快照、来源与使用说明。原始 `.pt` 权重、导出的 ONNX、TFLite、
DLA、人脸校准图片和板端原始输出均由 `.gitignore` 排除，留在本地、Ubuntu 89 和 Genio 720 板端。

冒烟通过只表示特征提取模型能在板端推理，不代表已完成人脸身份识别的精度、性能或产品验收。

## 当前验证结果

Genio 720 板端冒烟于 2026-09-24 完成，运行编号 `20260924T020219Z`。
三次 `neuronrt -m hw` 推理均生成 128 元素 INT8 特征，重复输入输出一致。
PyTorch 与 ONNX 单图输出最大绝对差约 `5.13e-6`，ONNX 与板端反量化输出余弦相似度约 `0.9901`。
详见 [冒烟记录](docs/smoke.md)。LFW 全量验证见[精度报告](docs/accuracy.md)，逐图 CLI 和常驻 Runtime 调用耗时见[性能报告](docs/benchmark.md)；机器人现场误识率、活体防护和完整应用链路性能仍待评估。
