# YOLOv5s · Genio 720

COCO 80 类目标检测，输入为 `1×3×640×640` RGB。模型来自 Ultralytics YOLOv5s v7.0；Qualcomm 页面仅作交付形式参考。Genio 720 已有板端交付证据，Genio 5100 尚未验证。来源、许可证和产物哈希见 [模型卡](model_card.md)，历史精度与性能见 [accuracy.md](docs/accuracy.md) 和 [benchmark.md](docs/benchmark.md)。

## 准备

先把官方 [yolov5s.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt) 手动放到 `models/yolov5s.pt`。YOLOv5 源码包与 MTK 补丁包也放在 `models/`，固定下载地址见 [source_url.txt](models/source_url.txt)。转换脚本只展开本地压缩包并应用补丁，不联网下载；Python 依赖由 89 的 Docker 环境管理。校准图片须在 89 的 `/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017/images`。

在 **89 宿主机**进入 `hhl_g720_8011`，然后在容器内执行一次：

```bash
docker exec -it hhl_g720_8011 bash
cd /data/users/hailong.he/github/mtk_models/models/perception/object_detection/yolov5s
bash deploy/prepare.sh
```

该入口依次完成本地源码准备、ONNX 与 INT8 TFLite 转换、`mdla5.3` DLA 编译。所用 `--suppress-output --disallow-bridge` 避开板端不支持的 EDPA 桥接。已有正确 DLA 时，无需为重跑测试再次转换。

## 板端测试

退出容器，在 **89 宿主机**的同一仓库目录执行。两种测试共用一个入口：

```bash
cd /data/users/hailong.he/github/mtk_models/models/perception/object_detection/yolov5s
bash deploy/test_board.sh smoke
EVAL_RUN_ID="$(date +%Y%m%d_%H%M%S)_yolov5s" bash deploy/test_board.sh full
```

| 模式 | 用途 | 前提与结果 |
| --- | --- | --- |
| `smoke` | `examples/input/` 中三张图片的 C++ 板端推理 | 结果在 `examples/output/`；只证明当前板端链路可运行。 |
| `full` | 5000 张 COCO val2017 的 C++ 板端推理、bbox AP 与耗时 | 92 须有 `/root/hailong.he/datasets/coco/val2017/images`、`annotations/instances_val2017.json` 和 `pycocotools`；使用每次唯一的 `EVAL_RUN_ID`，报告留在 `/root/hailong.he/open_models/yolov5s/eval/<run_id>/report/`。 |

默认模式为 `smoke`。两种模式都需要 `models/model_int8.dla`，并在 89 交叉编译板端 C++ 程序；`smoke` 还需要 `hhl_g720_8011` 容器来绘制检测框。脚本默认连接 `root@192.168.0.92`。若 SSH 提示主机密钥变化，应先通过可信渠道核对板端新指纹，再更新执行脚本用户的 `~/.ssh/known_hosts`；不要关闭主机密钥校验。

`full` 不自动回传、提交或清理报告。确认报告已备份并上传后，按 [全量复测工作流](../../../../docs/full_accuracy_board_workflow.md) 单独执行 `cleanup_full_accuracy.sh`。脚本与 Python/C++ 代码的目录说明见 [deploy/README.md](deploy/README.md)；历史三后端对照入口归档在 `archive/legacy_deploy/`。
