# YOLOv5s 模型卡

## 来源

- 对标页面：<https://huggingface.co/qualcomm/Yolo-v5>
- 上游实现：<https://github.com/ultralytics/yolov5>
- MTK 转换指南：<https://genio.mediatek.com/doc/iot-aihub/ai_hub/model_zoo/litert_analytical/YOLOv5s.html>
- 模型变体：YOLOv5s，不是 Qualcomm 页面当前展示的 YOLOv5-M。

## 许可证

Ultralytics YOLOv5 使用 AGPL-3.0。使用和再分发模型、修改代码或服务前必须评估许可证义务。

## 输入输出

输入为 NCHW RGB、FP32、范围 `[0, 1]`，固定形状 `1×3×640×640`。转换后的模型保留三个
原始检测头，后处理执行 sigmoid、anchor 解码、置信度过滤和 NMS。

## 文件校验

首次生成后在此记录来源提交号、权重 SHA-256 和全部转换产物 SHA-256。在未记录前不得标记为
“完整交付”。
