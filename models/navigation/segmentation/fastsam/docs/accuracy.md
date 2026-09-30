# 精度与一致性

当前状态：COCO val2017 全量 PyTorch、FP32 ONNX 与独立校准板端 DLA 已按统一协议完成。
主要指标为类别无关 segm AR@100。旧 DLA 使用了 16 张 val2017 图片校准，
与评测集重叠，因此旧板端 AR 仅作历史诊断。

## COCO val2017 全量精度

旧运行 `20260928_fastsam_full_v1` 在 5,000/5,000 张图片上得到板端类别无关
segm AP50:95 `0.0611437061`，[板端原始报告](../results/full_accuracy/20260928_fastsam_full_v1/summary.json)。
旧 FP32 ONNX 参考端 AP50:95 为 `0.0520981207`，[参考端原始报告](../results/reference_accuracy/coco_fp32_v1/summary.json)。
核查评测入口发现：板端使用置信度阈值 **0.001**，参考端使用 **0.4**；旧文档曾误写两端都是 0.4。
因此这两个数字不能作为同协议的精度差值，也不能说明量化提升。

对板端已保存的预测按 0.4 筛选，不重新运行模型，类别无关 segm AP50:95
仍约为 **0.060**，说明阈值不一致并非低 AP 的唯一原因。旧板端预测共
499,974 个实例，其中 263,571 个得分高于 0.4；旧参考端预测共 236,937 个。
目前通过单张 COCO 图片核对，原始 PyTorch 与 FP32 ONNX 的前 10 个实例
掩码 IoU 均大于 0.98。这仅验证该图片，不代替全量原始框架基线。

统一协议固定在 [`deploy/accuracy_protocol.json`](../deploy/accuracy_protocol.json)：
640×640 方形输入、置信度 0.001、NMS IoU 0.9、每图最多 100 个实例，
COCO val2017 的 80 类标注合并为一个 `object` 类。三端使用同一 5,000 图标注文件：

| 后端 | segm AR@100 | 预测实例 | 结果 |
| --- | ---: | ---: | --- |
| 官方 PyTorch FP32 | 0.391 | 499,881 | [结果摘要](../results/reference_accuracy/coco_pytorch_v2/summary.json) |
| FP32 ONNX | 0.390 | 499,882 | [结果摘要](../results/reference_accuracy/coco_fp32_v2/summary.json) |
| Genio 720 独立校准 INT8 DLA | 0.376384 | 499,956 | [板端全量报告](../results/full_accuracy/20260929_fastsam_imagenet100_full_v2/summary.json) |
| Genio 720 旧 INT8 DLA，校准集重叠 | 0.370 | 499,974 | [历史结果摘要](../results/full_accuracy/20260928_fastsam_full_v1/summary.json) |

新 DLA 在 5,000 张图上达到 0.376384，比同协议 ONNX 参考值约低 0.014。
板端 C++ 量化输入与 Python 参考量化输入在一张图上逐字节相同，且同一组板端原始张量经
C++、Python 后处理的 100 个掩码逐个一致；这限定了已验证的前后处理范围，
剩余差异仍需从 INT8 转换与硬件输出排查。官方 FastSAM 的目标候选生成评测报告 AR10、AR100、AR1000；
本表是 **segm AR@100**，与其 bbox 候选指标、FastSAM-x 权重和输入设置不同，
不能直接比较数值。该类别无关口径也不能与
标准 80 类 COCO segm AP、官方 FastSAM-x 的 1024 输入结果直接比较。

排查时对固定间隔抽取的 200 张 COCO 图比较了旧板端与 ONNX 的 20,000 个掩码：
板端掩码有 95.3% 能找到 IoU≥0.5 的 ONNX 掩码。单图原始张量核对显示 C++ 量化输入
与参考输入逐字节相同，同一组板端原始输出经 C++ 与 Python 后处理得到的 100 个掩码
逐个相同。新 DLA 使用 100 张 ImageNet 图片校准，与 5,000 张 COCO val2017 图片的
SHA-256 集合零重叠；200 张同图门禁的 ONNX / 旧 DLA / 新 DLA segm AR@100 分别为
0.408714 / 0.389318 / 0.393746。新 DLA 的 5,000 张结果见上表。

新 DLA 使用 ImageNet val 中 100 张互不重复图片校准，元数据 SHA-256 为
`722f5db18e5bd9f9b1b7d4090a89b2c36a047aef0e0511a3014b9b4a14c22bd7`，
DLA SHA-256 为 `6aa12c1c68d0eb13ee7669c83c36e838581a1a4ab917e2f73f37378edc13b7e5`。
COCO 评测图 5,000 张互不重复，校准图与评测图 SHA-256 交集为零；板端保存了
5,000 份逐图预测和完整预测 JSON。模型、协议、预测文件哈希均在[本次报告](../results/full_accuracy/20260929_fastsam_imagenet100_full_v2/summary.json)。

旧脚本还计算出同协议 PyTorch / ONNX / 板端 segm AP50:95 分别为
0.054621 / 0.054454 / 0.061144。这些数字保留在[原始 PyTorch 报告](../results/reference_accuracy/coco_pytorch_v2/summary.json)、
[ONNX 报告](../results/reference_accuracy/coco_fp32_v2/summary.json)和[板端报告](../results/full_accuracy/20260928_fastsam_full_v1/summary.json)
中供排查；类别无关目标候选的置信度排序会明显影响 AP，因此不作为此模型在总览中的主要精度结论。

数据位于板端 `/root/hailong.he/datasets/coco/val2017/`，共 5,000 张；标注文件 `annotations/instances_val2017.json` 的 SHA-256 为 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。数据从 89 的 `/data/users/hailong.he/datasets/coco/` 复制；原始数据来源见 [COCO 2017 下载页](https://cocodataset.org/#download)。

1. 导出时比较官方模型前向与拆分输出头后 CPU 解码结果,检查框、分数、系数和原型.
2. 在 89 的独立评测环境执行 ONNX CPU 推理,比较同图 PyTorch 与 ONNX 原始张量,
   FP32 使用 atol=1e-4、rtol=1e-3,失败时停止后续验证流程.
3. 在 92 用 C++ 执行 NPU 推理,回收原始输出后比较 ONNX 与 NPU 的逐头最大/平均绝对误差,
   以及固定阈值下按框 IoU>=0.5 贪心一对一匹配的掩码 IoU.
4. 同时报告两端实例数、匹配数和未匹配数; 禁止仅报告高匹配 IoU 隐藏漏检.

单图匹配 IoU 是转换一致性证据,不等于正式数据集 mAP.
FP32 原始头一致也不证明整个掩码后处理完全等价于所有版本的官方 Demo.
原始 PyTorch 与 FP32 ONNX 的全量总指标接近；逐图一致性、板端与参考端的逐图误差归因仍待分析。
校准与正式评价样本必须分离,不能用校准集结果代替正式测试精度.

2026-09-23 的 16 张 COCO 校准图与公共样例图的单图比较结果,
见 [板端冒烟报告](board_smoke_20260923.md).前 30 个实例各有 28 对匹配,
这些匹配实例的平均掩码 IoU 为 0.956245;两侧各有 2 个未匹配实例.
这是转换一致性指标,不是正式分割精度.
