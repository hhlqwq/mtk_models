# RTMPose Body2d

## 模型信息

```text
模型: RTMPose-Body2d
任务: WholeBody 人体姿态估计
输入: 1×3×256×192 RGB
输出: 133 个关节点的位置和置信度
设备: MediaTek Genio 720 EVK
当前状态: 板端已验证
```

RTMPose 是 top-down 姿态模型,只处理人体检测框.转换校准与板端 Demo 使用 COCO person
标注框模拟上游检测器输出,执行仿射裁剪、INT8 量化、板端推理和 SimCC 解码.正式
WholeBody 精度评测则固定使用 Faster R-CNN 检测框,具体协议见“正式评测数据”.完整图片应用
仍需额外的人体检测器；模型自身延迟和“检测器 + RTMPose"端到端延迟必须分别报告.

## 执行流程

```bash
cd /workspace/models/interaction/pose_detection/rtmpose_body2d
./deploy/download_original.sh
./deploy/convert.sh
./deploy/build.sh
./deploy/deploy_board.sh
# 正式 COCO-WholeBody 133 点精度评测.
./deploy/accuracy_board_cpp.sh
```

`download_original.sh` 支持复用 `models/` 中已有的官方 ZIP,保留 FP32 ONNX 外部权重,并
生成 MTK Converter 可读取的单文件兼容副本.`convert.sh` 默认使用 100 个 COCO person 框
校准；`build.sh` 固定使用
Genio 720 所需的 `mdla5.3 + --suppress-output + --disallow-bridge`；部署脚本使用两张不同
图片冒烟并采集 20 次预热、100 次连续推理、峰值内存及单次进程耗时.

## 交付状态

| 环节 | 状态 | 证据 |
| --- | --- | --- |
| Hugging Face 来源 | 已锁定 | `original/source_url.txt` |
| Qualcomm FP32 ONNX | 已完成 | `f2f68ac...c7768d`，外部权重 `bfe2b8c...922b3` |
| MTK 兼容 ONNX | 已验证等价 | 双输出 max/mean abs 均为 0 |
| MTK INT8 TFLite | 已完成 | `b996d17...3b89b` |
| DLA | 已完成 | `3d65b14...be5dc`，mdla5.3，无桥接 |
| 板端 Demo | 已完成 | 双图、双输出、133 点解码 |
| WholeBody AP | 正式数据已就绪,等待执行 | `docs/accuracy.md` |
| 板端性能 | 已完成 | 3.73232 ms/inf，262.3 FPS |

版本化验证证据见 `docs/board_validation_20260909.json`.

## 正式评测数据

正式精度评测固定使用 COCO-WholeBody V1.0 验证集标注、COCO val2017 原图和 MMPose
提供的 Faster R-CNN 人体检测框.不使用 `instances_val2017.json` 的 GT 人体框,也不替换为
其他检测器生成的人体框,确保结果可以与 MMPose/RTMPose 基准直接比较.

| 文件 | 下载来源 | NAS 路径 | SHA-256 |
| --- | --- | --- | --- |
| `coco_wholebody_val_v1.0.json` | [官方 Google Drive](https://drive.google.com/file/d/1N6VgwKnj8DeyGXCvp1eYgNbRmw6jdfrb/view?usp=sharing),[实际下载镜像](https://huggingface.co/datasets/msdkhairi/coco2017/resolve/main/coco_wholebody_val_v1.0.json?download=true) | `coco_val2017/annotations/` | `f8272e9c12f3a42457033ebc75da1167546edf1be5e2ffaf586d6ee97541ff6e` |
| `COCO_val2017_detections_AP_H_56_person.json` | [MMPose 官方 Google Drive 目录](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing),[实际下载镜像](https://huggingface.co/datasets/msdkhairi/coco2017/resolve/main/COCO_val2017_detections_AP_H_56_person.json?download=true) | `coco_val2017/person_detection_results/` | `53ba0ad8d0fd461c5a000cd90797fa8c39cd8c38cd125125c0412626ff592d59` |

NAS 数据集根目录为:

```text
\\192.168.0.68\Datasets\open_source\raw\coco\coco_val2017
```

89 服务器上的对应挂载目录为:

```text
/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017
```

当前文件检查结果:WholeBody 标注包含 5,000 张图和 11,004 个人标注；检测框文件包含
104,125 个 `category_id=1` 的人体检测框.两份文件均已通过 JSON 解析检查.

`accuracy_board_cpp.sh` 会生成锁定顺序的检测框清单,交叉编译并部署常驻 Neuron Runtime
C++ 推理器,先执行两个框的冒烟,再完成全部 104,125 个框的板端推理.指标阶段严格采用
MMPose 默认的 `bbox_keypoint` 重评分、0.2 关键点阈值和 0.9 WholeBody OKS-NMS,
最后通过 `xtcocotools` 分别计算 body、foot、face、left hand、right hand 和 wholebody
的 AP/AR.逐框进度和可续跑的 `processed_ids.txt` 用于观察长时间评测状态.

## 验证边界

- 双图板端冒烟用于确认 DLA 可运行、输出完整且不同输入不会得到完全相同的旧缓冲结果.
- `instances_val2017.json` 只用于转换校准和 Demo 框输入,不作为正式 WholeBody AP 的人体框来源.
- 正式 133 点 WholeBody AP 的数据与人体框协议已经锁定,但完整板端评测尚未执行.
