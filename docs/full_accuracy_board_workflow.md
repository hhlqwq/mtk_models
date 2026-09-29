# Genio 720 全量精度复测工作流

本工作流用于板端重刷镜像后的重新验证。**一次只选择并测试一个模型**,不得使用循环或总入口连续启动六个模型；前一个模型的报告应先检查并按需清理,再启动下一个。下述六个模型各有两个独立入口：

1. `deploy/run_full_accuracy.sh`：用户在 Ubuntu89 宿主机手动启动一次。89 只负责模型/程序编译和部署调度；92 负责数据准备、C++ NPU 推理和 Python 指标计算。新镜像全量测试不运行 89 端精度基线。成功后**保留模型、逐样本检查点和报告**。
2. `deploy/cleanup_full_accuracy.sh`：用户从 92 手动取走 `report/`、上传到 Git 并核验后，另一次手动执行。它校验报告完整性和 SHA-256，保留小体积报告到 92 的 `results/<run_id>/`，删除该 run 的 `eval/<run_id>/` 大文件、`models/<run_id>/` 专属模型副本及 `/root/hailong.he/datasets/<model>/<run_id>/` 专属缓存。不会删除共享原始数据集或官方 `/root/hailong.he/MTK_G720_DLA`。

**任何脚本都不执行 Git 提交、推送或结果回传。** 在用户上传 Git 之前，92 是结果的唯一保存位置。下一次刷机前，务必先自行备份结果。清理需要显式设置 `CONFIRM_RESULTS_UPLOADED=1`，脚本仅信任用户确认，不会替用户查询 Git 远端。

## 板端目录与前提

- 本次专属模型：`/root/hailong.he/open_models/<model>/models/<run_id>/`；运行目录：`/root/hailong.he/open_models/<model>/eval/<run_id>/`，测试完成时报告在其 `report/` 子目录。共享旧模型文件不会被清理脚本碰触。
- 清理后保留报告：`/root/hailong.he/open_models/<model>/results/<run_id>/`。
- 共享数据集根目录：`/root/hailong.he/datasets/`。脚本不下载、不解压、不移动共享原始数据。
- 92 的系统时间、运行库和指标计算所需 Python 依赖须先正确配置。2026-09-28 已校准时钟并准备本轮完整数据和板端指标依赖；各数据集的来源、SHA-256、数量和正式结果记录在对应模型的 `docs/accuracy.md`，耗时记录在 `docs/benchmark.md`。正式入口仍逐次检查时间、数据和依赖。

板端已安装 `pycocotools==2.0.10`、`opencv-python-headless==4.10.0.84`、`tiktoken==0.11.0`、`regex==2025.9.18`、`more-itertools==10.7.0`。本轮离线安装包由清华 PyPI 镜像获取，精确地址与 SHA-256 如下；89 端 LFW 整理另用 `pyarrow==17.0.0`。这些是环境资源记录，模型结果仍以各模型文档为准。

| 包 | 下载地址 | SHA-256 |
| --- | --- | --- |
| `pycocotools 2.0.10` | <https://pypi.tuna.tsinghua.edu.cn/packages/29/d5/b17bb67722432a191cb86121cda33cd8edb4d5b15beda43bc97a7d5ae404/pycocotools-2.0.10-cp312-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl> | `075788c90bfa6a8989d628932854f3e32c25dac3c1bf7c1183cefad29aee16c8` |
| `opencv-python-headless 4.10.0.84` | <https://pypi.tuna.tsinghua.edu.cn/packages/91/61/f838ce2046f3ec3591ea59ea3549085e399525d3b4558c4ed60b55ed88c0/opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl> | `46071015ff9ab40fccd8a163da0ee14ce9846349f06c6c8c0f2870856ffa45db` |
| `tiktoken 0.11.0` | <https://pypi.tuna.tsinghua.edu.cn/packages/65/8e/c769b45ef379bc360c9978c4f6914c79fd432400a6733a8afc7ed7b0726a/tiktoken-0.11.0-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl> | `6ef72aab3ea240646e642413cb363b73869fed4e604dcfd69eec63dc54d603e8` |
| `regex 2025.9.18` | <https://pypi.tuna.tsinghua.edu.cn/packages/ee/66/243edf49dd8720cba8d5245dd4d6adcb03a1defab7238598c0c97cf549b8/regex-2025.9.18-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl> | `300e25dbbf8299d87205e821a201057f2ef9aa3deb29caa01cd2cac669e508d5` |
| `more-itertools 10.7.0` | <https://pypi.tuna.tsinghua.edu.cn/packages/2b/9f/7ba6f94fc1e9ac3d2b853fdff3035fb2fa5afbed898c4a72b8a020610594/more_itertools-10.7.0-py3-none-any.whl> | `d43980384673cb07d2f7d2d918c616b30c659c089ee23953f601d6609c67510e` |
| `pyarrow 17.0.0` | <https://pypi.tuna.tsinghua.edu.cn/packages/f1/c4/9625418a1413005e486c006e56675334929fad864347c5ae7c1b2e7fe639/pyarrow-17.0.0-cp312-cp312-manylinux_2_28_x86_64.whl> | `b0c6ac301093b42d34410b187bba560b17c0330f64907bfa4f7f2444b0cf9b` |

另有两个下载后未安装的候选包，均保留在 89 的 `/tmp/hailongcodex/20260928/`：

| 候选包 | 下载地址 | SHA-256 |
| --- | --- | --- |
| `opencv-python-headless 5.0.0.93` | <https://pypi.tuna.tsinghua.edu.cn/packages/ec/78/afca939f40ffe2b2380bfa86f812b2f7d4acc5a27b27dc41b49cad7ce7b4/opencv_python_headless-5.0.0.93-cp37-abi3-manylinux2014_aarch64.manylinux_2_17_aarch64.whl> | `10818d91510e05c04568ae12b5cd120779c70c01bf897b001a6221fe430df80f` |
| `regex 2026.9.10` | <https://pypi.tuna.tsinghua.edu.cn/packages/89/51/3fb5fe0d32f4cf0bc982286722c729a8d6f522d2fa2d5d14a702d9fc87f8/regex-2026.9.10-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl> | `4c66d54042a14a503907d81861b8a5235e6d1f03d4fbc1d8767f652eaf957ac1` |

| 模型 | 板端数据与协议 | 92 所需额外依赖 | 指标口径 |
| --- | --- | --- | --- |
| `whisper_tiny` | `datasets/librispeech/test-clean`；必须为 2620 条 | 指标阶段 OpenAI Whisper tokenizer；音频解码需 ffmpeg | C++ 解码 FLAC、生成 Mel 并执行双 DLA 推理；Python 只计算 LibriSpeech `test-clean` WER；旧 AISHELL-1 CER 仅保留为历史证据 |
| `yolov5s` | `datasets/coco/val2017/images` 5000 张及 `annotations/instances_val2017.json` | pycocotools | COCO bbox AP；常驻 C++ Runtime 耗时 |
| `rtmpose_body2d` | 同一 5000 张 COCO 图片、WholeBody 标注和官方人体检测框 JSON | xtcocotools、NumPy | 104125 个框的 NPU WholeBody AP/AR；C++ 耗时 |
| `vit_base_patch16_224` | `datasets/imagenet/val` 50000 张和 `val_labels_0based.txt` | 板端 OpenCV C++ Runtime | C++ 常驻 NPU 推理、Python 计算 Top-1/Top-5；另列排除 PTQ 校准区间的 49900 张；耗时仅计 Runtime 推理调用 |
| `yoloworld_xl` | COCO val2017 5000 张及 bbox 标注 | C++ ONNX Runtime 1.20.2 Neuron EP、指标阶段 pycocotools | C++ 常驻会话完成推理；Python 只计算 COCO bbox AP；profiling 必须有 Neuron 节点 |
| `fastsam` | COCO val2017 5000 张及实例分割标注 | 指标阶段 OpenCV、NumPy、pycocotools | **类别无关** COCO segm AP：将 GT 的 80 类合并为 `object`；不可与标准 80 类 segm AP 横比。shell 逐图调用 C++，Python 只编码掩码并计算 AP |

代码迁移状态：六个模型的新入口均以 C++ 完成板端预处理和推理、Python 仅用于指标阶段。Whisper-Tiny 的 LibriSpeech `test-clean`、FastSAM 的 COCO val2017 类别无关分割及 YOLO-World XL 的 COCO val2017 bbox 全量测试已在 92 完成，报告分别保存在模型的 `results/full_accuracy/` 目录。RTMPose 与 ViT 的新入口仍需按各自运行报告核对，不可把交叉编译或静态检查当作板端验证。旧的 Python 推理脚本只供历史复现，不由本工作流调用。

所有模型使用新镜像对应的全新 `EVAL_RUN_ID`。**只验完整的官方 test/val split，不抽取 500、1000 张等部分子集；覆盖数量不符就不能生成 `complete` 报告。** 中断或前置检查失败时不自动清理；先查看该 run 的日志和状态。不要把历史旧镜像结果复用为新镜像验证。`run_full_accuracy.sh` 只处理全量精度；单图冒烟脚本仍可分开运行。

## 手动执行

在 Ubuntu89 宿主机的仓库根目录执行，每次只跑一个模型；不要在容器内部启动总入口。例如：

```bash
cd /data/users/hailong.he/github/mtk_models
EVAL_RUN_ID=20260923_yolov5s_new_bsp_v1 \
  bash models/perception/object_detection/yolov5s/deploy/run_full_accuracy.sh
```

其他模型使用相同入口。Whisper-Tiny、FastSAM 和 YOLO-World XL 已完成本轮全量评测；其他模型以各自报告的 `status`、覆盖数量和哈希为准：

```text
models/audio/stt/whisper_tiny/deploy/run_full_accuracy.sh
models/interaction/pose_detection/rtmpose_body2d/deploy/run_full_accuracy.sh
models/perception/image_classification/vit_base_patch16_224/deploy/run_full_accuracy.sh
models/perception/object_detection/yoloworld_xl/deploy/run_full_accuracy.sh
models/navigation/segmentation/fastsam/deploy/run_full_accuracy.sh
```

报告成功条件是 `report/summary.json` 的 `status` 为 `complete`，且数量、数据集和模型身份与本次 run 相符。请在 92 上先查看报告与日志；从 92 手动取走整个 `report/`，存入仓库对应模型的 `results/full_accuracy/<run_id>/`，自行 `git add`、提交和推送。模型权重、DLA、完整数据集、逐图原始输出不放进普通 Git。报告中的哈希记录可用于核对板端原始产物。若报告超过 20 MiB，清理入口会拒绝执行，需先检查报告是否混入大文件。

手动上传并确认后，**另起一次命令**清理对应 run。例如：

```bash
cd /data/users/hailong.he/github/mtk_models
EVAL_RUN_ID=20260923_yolov5s_new_bsp_v1 CONFIRM_RESULTS_UPLOADED=1 \
  bash models/perception/object_detection/yolov5s/deploy/cleanup_full_accuracy.sh
```

清理入口只删除明确匹配 `/root/hailong.he/open_models/<model>/eval/<run_id>`、`/root/hailong.he/open_models/<model>/models/<run_id>` 和 `/root/hailong.he/datasets/<model>/<run_id>` 的内容；报告移动到同模型 `results/<run_id>`。失败或不完整的测试没有 `complete` 报告，不能清理。系统镜像、共享数据集、其他模型、其他 run 和官方 Model Zoo 目录都不在清理范围。

## 结果解释

六个入口均在 `report/system.txt` 记录板端系统和 Runtime 身份,并在 `summary.json` 写入 run ID、模型、数据集和完整状态；各模型另保留模型/数据哈希与指标。不能直接以旧镜像结果标注“新镜像已验证”。YOLOv5s、RTMPose、ViT 和 YOLO-World 的板端 C++ 常驻推理耗时口径与 FastSAM 每图重载模型不同；跨模型速度比较前必须先统一预热、线程、输入大小和测时范围。

2026-09-28 另新增 [Depth Anything V2 Small](../models/navigation/single_camera_depth/depth_anything_v2_small/README.md) 的 DA-2K 点对精度入口与 [MobileFaceNet](../models/interaction/face_recognition/mobilefacenet/README.md) 的 LFW 十折验证入口。两者采用逐图 `neuronrt` CLI，报告的 `cli_wall_ms` 包含进程启动和模型加载，不与上述常驻 C++ Runtime 的纯推理耗时直接比较。MobileFaceNet 使用未对齐 LFW 图片的固定缩放协议，准确率不能与上游对齐人脸的结果直接比较。两者不使用本工作流六模型的清理入口；运行目录和原始数据保留到用户审阅。
