# 精度报告

## 当前状态

正式 COCO bbox mAP 尚未执行,不能用三张公开样例替代.

三张公开图片的混合 Neuron EP 推理已通过 CPU 一致性门禁.固定配置为完整 opset 13 模型、
`NEURON_FLAG_USE_FP16=1` 和 `NEURON_FLAG_MIN_GROUP_SIZE=100`：

| 指标 | 结果 | 门槛 |
| --- | ---: | ---: |
| 三图检测数量 | CPU/Neuron 均为 `10/13/4` | 必须相同 |
| 类别及排序 | 全部相同 | 必须相同 |
| 最大置信度差 | `0.0016256571` | `≤0.01` |
| 最大框坐标差 | `0.0419921875 px` | `≤0.5 px` |

`deploy/inference_demo/compare_results.py` 实施上述门禁.当最小子图为官方 benchmark 默认的
`0` 时,分类 logits 明显漂移并导致三图各 300 个检测；该快速配置仅能说明执行和性能,
不能用于检测结果.

兼容模型已通过 `deploy/verify_onnx_equivalence.py`,证明 opset 11 到 opset 13 转换没有
改变六个 CPU EP 输出；Raw DFL 重建最大绝对差小于 `3.82e-6`.

正式精度计划使用板端已有 COCO val2017 5,000 张图片及 `instances_val2017.json`,固定
score threshold、IoU threshold、最大检测数和 COCO 80 类映射.只有完整处理清单、预测
JSON、模型/数据集/代码哈希和 pycocotools 指标齐全后,才更新本报告.
