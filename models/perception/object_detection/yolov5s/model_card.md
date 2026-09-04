# YOLOv5s 模型卡

## 来源

- 对标页面：<https://huggingface.co/qualcomm/Yolo-v5>
- 上游实现：<https://github.com/ultralytics/yolov5>
- 固定源码提交：`485da42273839d20ea6bdaf142fd02c1027aba61`
- YOLOv5s v7.0 权重：<https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt>
- MTK 转换指南：<https://genio.mediatek.com/doc/iot-aihub/ai_hub/model_zoo/litert_analytical/YOLOv5s.html>
- 模型变体：YOLOv5s，不是 Qualcomm 页面当前展示的 YOLOv5-M。

## 许可证

Ultralytics YOLOv5 使用 AGPL-3.0。使用和再分发模型、修改代码或服务前必须评估许可证义务。

## 输入输出

输入为 NCHW RGB、FP32、范围 `[0, 1]`，固定形状 `1×3×640×640`。转换后的模型保留三个
原始检测头，后处理执行 sigmoid、anchor 解码、置信度过滤和 NMS。

## 文件校验

本机权重 `models/yolov5s.pt` 的大小为 14,808,437 bytes，SHA-256 为
`8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee`。转换后还必须记录
TorchScript、ONNX、INT8 TFLite 和 DLA 的 SHA-256；未记录完整前不得标记为“完整交付”。
