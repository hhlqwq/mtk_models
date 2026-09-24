# Genio 720 全量精度复测工作流

本工作流用于板端重刷镜像后的重新验证。**一次只选择并测试一个模型**,不得使用循环或总入口连续启动六个模型；前一个模型的报告应先检查并按需清理,再启动下一个。每个模型有两个独立入口：

1. `deploy/run_full_accuracy.sh`：用户在 Ubuntu89 宿主机手动启动一次。89 只负责模型/程序编译和部署调度；92 负责数据准备、C++ NPU 推理和 Python 指标计算。新镜像全量测试不运行 89 端精度基线。成功后**保留模型、逐样本检查点和报告**。
2. `deploy/cleanup_full_accuracy.sh`：用户从 92 手动取走 `report/`、上传到 Git 并核验后，另一次手动执行。它校验报告完整性和 SHA-256，保留小体积报告到 92 的 `results/<run_id>/`，删除该 run 的 `eval/<run_id>/` 大文件、`models/<run_id>/` 专属模型副本及 `/root/hailong.he/datasets/<model>/<run_id>/` 专属缓存。不会删除共享原始数据集或官方 `/root/hailong.he/MTK_G720_DLA`。

**任何脚本都不执行 Git 提交、推送或结果回传。** 在用户上传 Git 之前，92 是结果的唯一保存位置。下一次刷机前，务必先自行备份结果。清理需要显式设置 `CONFIRM_RESULTS_UPLOADED=1`，脚本仅信任用户确认，不会替用户查询 Git 远端。

## 板端目录与前提

- 本次专属模型：`/root/hailong.he/open_models/<model>/models/<run_id>/`；运行目录：`/root/hailong.he/open_models/<model>/eval/<run_id>/`，测试完成时报告在其 `report/` 子目录。共享旧模型文件不会被清理脚本碰触。
- 清理后保留报告：`/root/hailong.he/open_models/<model>/results/<run_id>/`。
- 共享数据集根目录：`/root/hailong.he/datasets/`。脚本不下载、不解压、不移动共享原始数据。
- 92 的系统时间、运行库和指标计算所需 Python 依赖须先正确配置。2026-09-23 的只读核对发现 92 时间仍显示 `2025-05-31`，数据集目录不完整，且缺少部分指标包；当前环境无法立即运行全部正式评测。所有正式入口先检查 89/92 时间差不超过 10 分钟，再检查所需数据和依赖；本次代码修改没有安装包或下载数据。

| 模型 | 板端数据与协议 | 92 所需额外依赖 | 指标口径 |
| --- | --- | --- | --- |
| `whisper_tiny` | `datasets/librispeech/test-clean`；必须为 2620 条 | 指标阶段 OpenAI Whisper tokenizer；音频解码需 ffmpeg | C++ 解码 FLAC、生成 Mel 并执行双 DLA 推理；Python 只计算 LibriSpeech `test-clean` WER；旧 AISHELL-1 CER 仅保留为历史证据 |
| `yolov5s` | `datasets/coco/val2017/images` 5000 张及 `annotations/instances_val2017.json` | pycocotools | COCO bbox AP；常驻 C++ Runtime 耗时 |
| `rtmpose_body2d` | 同一 5000 张 COCO 图片、WholeBody 标注和官方人体检测框 JSON | xtcocotools、NumPy | 104125 个框的 NPU WholeBody AP/AR；C++ 耗时 |
| `vit_base_patch16_224` | `datasets/imagenet/val` 50000 张和 `val_labels_0based.txt` | 板端 OpenCV C++ Runtime | C++ 常驻 NPU 推理、Python 计算 Top-1/Top-5；另列排除 PTQ 校准区间的 49900 张；耗时仅计 Runtime 推理调用 |
| `yoloworld_xl` | COCO val2017 5000 张及 bbox 标注 | C++ ONNX Runtime 1.20.2 Neuron EP、指标阶段 pycocotools | C++ 常驻会话完成推理；Python 只计算 COCO bbox AP；profiling 必须有 Neuron 节点 |
| `fastsam` | COCO val2017 5000 张及实例分割标注 | 指标阶段 OpenCV、NumPy、pycocotools | **类别无关** COCO segm AP：将 GT 的 80 类合并为 `object`；不可与标准 80 类 segm AP 横比。shell 逐图调用 C++，Python 只编码掩码并计算 AP |

代码迁移状态：六个模型的新入口均以 C++ 完成板端预处理和推理、Python 仅用于指标阶段；RTMPose 检测框清单生成器、ViT、Whisper 音频准备程序、YOLO-World 的 ONNX Runtime C API 程序已通过交叉编译语法检查，但这些新流程均尚未在 92 实测。不要把静态检查当作板端验证。旧的 Python 推理脚本只供历史复现,不由本工作流调用。

所有模型使用新镜像对应的全新 `EVAL_RUN_ID`。**只验完整的官方 test/val split，不抽取 500、1000 张等部分子集；覆盖数量不符就不能生成 `complete` 报告。** 中断或前置检查失败时不自动清理；先查看该 run 的日志和状态。不要把历史旧镜像结果复用为新镜像验证。`run_full_accuracy.sh` 只处理全量精度；单图冒烟脚本仍可分开运行。

## 手动执行

在 Ubuntu89 宿主机的仓库根目录执行，每次只跑一个模型；不要在容器内部启动总入口。例如：

```bash
cd /data/users/hailong.he/github/mtk_models
EVAL_RUN_ID=20260923_yolov5s_new_bsp_v1 \
  bash models/perception/object_detection/yolov5s/deploy/run_full_accuracy.sh
```

完成板端运行验证后，其他模型也使用相同入口。目前这些入口是代码层面的实现，
板端可运行性和实际精度仍需用户按完整数据集手动验证：

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

六个入口均在 `report/system.txt` 记录板端系统和 Runtime 身份,并在 `summary.json` 写入 run ID、模型、数据集和完整状态；各模型另保留模型/数据哈希与指标。不能直接以旧镜像结果标注“新镜像已验证”。YOLOv5s、RTMPose、ViT 和 YOLO-World 的板端 C++ 常驻推理耗时口径与 FastSAM 每图重载模型不同；跨模型速度比较前必须先统一预热、线程、输入大小和测时范围。MobileFaceNet 的正式数据集和指标协议尚未确定,不属于本轮六模型复测。
