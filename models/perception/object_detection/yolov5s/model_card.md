# YOLOv5s 模型卡

## 来源

- 对标页面：<https://huggingface.co/qualcomm/Yolo-v5>
- 上游实现：<https://github.com/ultralytics/yolov5>
- 固定源码提交：`485da42273839d20ea6bdaf142fd02c1027aba61`
- YOLOv5s v7.0 权重：<https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt>
- MTK 转换指南：<https://genio.mediatek.com/doc/iot-aihub/ai_hub/model_zoo/litert_analytical/YOLOv5s.html>
- 模型变体：YOLOv5s,不是 Qualcomm 页面当前展示的 YOLOv5-M.

## 许可证

Ultralytics YOLOv5 使用 AGPL-3.0.使用和再分发模型、修改代码或服务前必须评估许可证义务.

## 输入输出

输入为 NCHW RGB、FP32、范围 `[0, 1]`,固定形状 `1×3×640×640`.转换后的模型保留三个
原始检测头,后处理执行 sigmoid、anchor 解码、置信度过滤和 NMS.

## 文件校验

以下哈希来自 2026-09-07 在 89 编译机生成并用于 Genio 720 实测的同一组产物：

| 文件 | 大小（bytes） | SHA-256 |
| --- | ---: | --- |
| `models/yolov5s.pt` | 14,808,437 | `8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee` |
| `models/yolov5s.torchscript` | 29,215,556 | `cbef4136e07e79b59e26659a3cea4bb7c5cc7e7a2b6fb80e4e79409b9f0818bd` |
| `models/model_fp32.onnx` | 28,942,238 | `a2e2f888f94ad07a93591de91f0e010792458b4c737795a13118fef9180f9778` |
| `models/model_int8.tflite` | 7,723,624 | `49212dfd6d02842abc690871f6580578f4573487bc7b34882434b407746b3fee` |
| `models/model_int8.dla` | 7,655,285 | `cf5b66c3fc1c24c9ef1d5c579d20f8d145cbd15d3a5be874b114f7f270c824c6` |

模型文件默认不进入普通 Git 历史.重新转换或编译后必须重新生成并核对哈希,不能沿用本表.

## 最终交付运行

Genio 720 最终交付运行 ID 为 `20260908_cpp_delivery_v3`,运行提交为
`1dcadbe4a5dec82b2e95efd3e0ffea90354a6bfc`.板端二进制 SHA-256 为
`daf756b4553c8ba07d36e0fa3761d9e6ce60651871b5a9a4d633aa0bdfffffaf`,COCO 标注
SHA-256 为 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`,
5000 张图片哈希清单的 SHA-256 为
`1bc0eca834162aace33aac3b67e9e5dc47a12aadc7f5e92108b4134161358f48`.详细清单见
`examples/output/board_cpp_accuracy/20260908_cpp_delivery_v3/`.
