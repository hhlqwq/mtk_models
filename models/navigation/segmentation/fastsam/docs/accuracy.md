# 精度与一致性

当前状态: 已完成同图 PyTorch 与板端 C++ NPU 冒烟比较，以及 COCO val2017 全量类别无关实例分割评测。

## COCO val2017 全量精度

运行编号 `20260928_fastsam_full_v1`，Genio 720 板端处理 5,000/5,000 张图片，报告状态为 `complete`。将标注中的 80 个类别合并为 `object`，类别无关 segm AP50:95 为 `0.0611437061`，AP50 为 `0.1056471507`，[板端报告](../results/full_accuracy/20260928_fastsam_full_v1/summary.json)。同 5,000 张、同类别无关 GT、置信度 0.4、NMS IoU 0.9、最多 100 个实例的 FP32 ONNX 参考端 AP50:95 为 `0.0520981207`，AP50 为 `0.0925646343`，[参考报告](../results/reference_accuracy/coco_fp32_v1/summary.json)。两端的掩码后处理分别由板端 C++ 和参考端 Python 实现，聚合 AP 的差异不能单独解释为量化精度提升；需逐图核查预处理、掩码还原和候选排序。该类别无关口径不能与标准 80 类 COCO segm AP 直接比较。

数据位于板端 `/root/hailong.he/datasets/coco/val2017/`，共 5,000 张；标注文件 `annotations/instances_val2017.json` 的 SHA-256 为 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。数据从 89 的 `/data/users/hailong.he/datasets/coco/` 复制；原始数据来源见 [COCO 2017 下载页](https://cocodataset.org/#download)。校准图片与正式评价图片的使用边界仍需以运行清单核对。

1. 导出时比较官方模型前向与拆分输出头后 CPU 解码结果,检查框、分数、系数和原型.
2. 在 89 的独立评测环境执行 ONNX CPU 推理,比较同图 PyTorch 与 ONNX 原始张量,
   FP32 使用 atol=1e-4、rtol=1e-3,失败时停止后续验证流程.
3. 在 92 用 C++ 执行 NPU 推理,回收原始输出后比较 ONNX 与 NPU 的逐头最大/平均绝对误差,
   以及固定阈值下按框 IoU>=0.5 贪心一对一匹配的掩码 IoU.
4. 同时报告两端实例数、匹配数和未匹配数; 禁止仅报告高匹配 IoU 隐藏漏检.

单图匹配 IoU 是转换一致性证据,不等于正式数据集 mAP.
FP32 原始头一致也不证明整个掩码后处理完全等价于所有版本的官方 Demo.
原始官方 PyTorch 端与当前 FP32 ONNX 的全量逐图一致性、板端与参考端的逐图误差归因仍待分析。
校准与正式评价样本必须分离,不能用校准集结果代替正式测试精度.

2026-09-23 的 16 张 COCO 校准图与公共样例图的单图比较结果,
见 [板端冒烟报告](board_smoke_20260923.md).前 30 个实例各有 28 对匹配,
这些匹配实例的平均掩码 IoU 为 0.956245;两侧各有 2 个未匹配实例.
这是转换一致性指标,不是正式分割精度.
