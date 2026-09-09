# RTMPose Body2d

## 模型信息

```text
模型: RTMPose-Body2d
任务: WholeBody 人体姿态估计
输入: 1×3×256×192 RGB
输出: 133 个关节点的位置和置信度
设备: MediaTek Genio 720 EVK
当前状态: 环境建设中（OpenMMLab 上游版本和权重待锁定）
```

> 迁移说明：本目录现有模型产物和结果来自 Qualcomm v0.61.0 预导出 ONNX,仅保留为
> 历史工程证据.正式交付将从 OpenMMLab MMPose 官方配置和权重自行导出 ONNX；固定
> 版本和权重未核验前,旧结果不再计入当前交付状态.

RTMPose 是 top-down 姿态模型,只处理人体检测框.转换校准与板端 Demo 使用 COCO person
标注框模拟上游检测器输出,执行仿射裁剪、INT8 量化、板端推理和 SimCC 解码.正式
WholeBody 精度评测则固定使用 Faster R-CNN 检测框,具体协议见“正式评测数据”.完整图片应用
仍需额外的人体检测器；模型自身延迟和“检测器 + RTMPose"端到端延迟必须分别报告.

## 目标执行流程

```bash
cd /workspace/models/interaction/pose_detection/rtmpose_body2d
# 先锁定 MMPose 固定版本、官方配置、权重、许可证和 SHA-256.
./deploy/download_original.sh
./deploy/convert.sh
./deploy/build.sh
./deploy/deploy_board.sh
# 正式 COCO-WholeBody 133 点精度评测.
./deploy/accuracy_board_cpp.sh
```

历史 `download_original.sh` 曾复用 Qualcomm ONNX 归档,现已停用.新流程必须从锁定的
MMPose 原始框架权重自行导出 ONNX,再根据实际计算图实现 MTK 兼容转换.`convert.sh`
默认使用 100 个 COCO person 框校准；`build.sh` 固定使用
Genio 720 所需的 `mdla5.3 + --suppress-output + --disallow-bridge`；部署脚本使用两张不同
图片冒烟并采集 20 次预热、100 次连续推理、峰值内存及单次进程耗时.

## 交付状态

| 环节 | 状态 | 证据 |
| --- | --- | --- |
| OpenMMLab 官方上游 | 待锁定 | 固定版本、配置、权重、许可证和 SHA-256 均不得猜测 |
| 原始框架基线 | 待执行 | 必须使用最终锁定的开源权重 |
| 自行导出 ONNX | 待执行 | 不得复用 Qualcomm 预导出 ONNX |
| MTK 兼容 ONNX | 待执行 | 需要根据新 ONNX 重新分析图结构 |
| MTK INT8 TFLite / DLA | 待执行 | 旧产物仅作历史对照 |
| 板端 Demo、WholeBody AP 和性能 | 待执行 | 新模型必须使用新的运行 ID 完整复测 |

历史版本化验证证据见 `docs/board_validation_20260909.json`.

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

正式运行 `20260909_wholebody_int8_v2` 已在 Genio 720 完成全部 104,125 个框；OKS-NMS
后保留 89,565 个结果，WholeBody AP 为 0.4369、AR 为 0.5646.板端常驻 C++ 流程的
平均预处理、NPU、后处理耗时分别为 2.7740 ms、3.8500 ms、1.0524 ms，峰值 RSS
为 35,756 KB.完整分部指标和结果哈希见 `docs/accuracy.md`.

以下输入约束只适用于历史 Qualcomm 图,不能直接套用于新的 MMPose 导出图.历史模型输入
保持原图定义的像素量纲:BGR FP32 `[0,255]`,再由 INT8 输入量化参数
转换为板端张量.不得在图外先除以 255；ONNX 图内的 ImageNet mean/std 常量同样采用
`[0,255]` 量纲.

## 验证边界

- 双图板端冒烟用于确认 DLA 可运行、输出完整且不同输入不会得到完全相同的旧缓冲结果.
- `instances_val2017.json` 只用于转换校准和 Demo 框输入,不作为正式 WholeBody AP 的人体框来源.
- 正式 133 点 WholeBody AP 已按锁定数据和人体框协议在板端完整执行；PyTorch 与 FP32
  ONNX 的同协议全量 AP 尚未单独执行，不与板端 INT8 指标混写.
