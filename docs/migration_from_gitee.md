# Gitee 旧工程迁移记录

来源工程：`D:\code\gitee\mtk`.

## 已确认可复用内容

- YOLOv5n 和 YOLOv8s 的 PyTorch 到 INT8 TFLite、DLA 编译流程.
- `neuronrt -m hw` 板端调用方式.
- YOLO 检测头解码、NMS 和结果绘制逻辑.
- 原始 PyTorch、PC TFLite、板端 DLA 三路 COCO 指标对比方法.
- 已记录的 Genio 720 YOLOv5n 和 YOLOv8s 性能结果.

## 迁移约束

- 旧仓库保持只读,不删除、不改写历史.
- 旧模型结果必须标记模型变体和运行时版本,不直接套用到 YOLOv5s.
- 新仓库按单模型交付结构拆分,不继续使用单个超长 `*_build.py` 聚合所有职责.
- 旧权重和中间产物不直接写入普通 Git 历史.

## 当前进度

基础环境、交付结构和板端验收口径已迁入.YOLOv5 的转换、部署、后处理和精度评测逻辑
将在 YOLOv5s 首次闭环过程中完成重构；YOLOv8s 作为后续模型保留,不计入首批三个模型.
