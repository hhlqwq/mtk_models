# 精度报告

## 当前状态

COCO val2017 全量 bbox mAP 已完成。运行编号 `20260928_yoloworld_coco_full_v3`，Genio 720 使用正确性优先的混合 Neuron/CPU EP 处理 5,000/5,000 张图片，bbox AP50:95 `0.4729515582`，AP50 `0.6368820482`，[板端报告](../results/full_accuracy/20260928_yoloworld_coco_full_v3/summary.json)。同一 5,000 张、80 类映射、分数阈值 0.001、NMS IoU 0.65、最多 300 框的 FP32 ONNX CUDA 优先参考端 AP50:95 `0.4729530726`，AP50 `0.6370186653`，[参考报告](../results/reference_accuracy/coco_fp32_v1/summary.json)。两端 AP50:95 相差约 `0.0000015`，但该板端会话仍包含 CPU fallback，不能描述为全图纯 NPU 精度。标注 SHA-256 为 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`，板端路径 `/root/hailong.he/datasets/coco/val2017/`；原始数据来源见 [COCO 2017 下载页](https://cocodataset.org/#download)。

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

全量运行保存完整处理清单、预测 JSON、模型/数据集/代码哈希和 pycocotools 指标。前两次运行因 C API provider 名称错误在第一张图片前退出，正式结果为 `v3`；注册名按 [MediaTek 文档](https://genio.mediatek.com/doc/iot-aihub/ai_hub/supported_os/yocto/onnxruntime/accelerating_ort.html)使用 `Neuron`。
