# Genio 720 汇总报告

模型完成板端验证后，由公共报告工具从各模型 `model.yaml`、`accuracy.md` 和 `benchmark.md`
生成汇总表。当前不得将 MTK 官网参考值写成本项目实测值。

## 当前状态

| 模型 | 状态 | 已验证结果 | 待完成 |
| --- | --- | --- | --- |
| YOLOv5s | 完整交付 | COCO val2017 三后端精度；板端 C++ 全流程 AP 0.3586；稳态端到端平均 33.662 ms、P95 38.009 ms；峰值 RSS 33,224 KiB | 无 |
| ViT-Base Patch16 224 | 板端已验证 | 1000 张 FP32/NPU Top-1 一致率 92.3%；纯 NPU 53.3903 ms；峰值 RSS 94,604 KB | 可靠 ImageNet 标签映射与绝对 Top-1/Top-5 |
