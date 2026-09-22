# 输入图片

优先复用仓库 YOLOv5s 的 `examples/input/public` 中已有 CC0 图片.
使用时保留该目录的 ASSET_LICENSE.md 与 PROMPTS.md; 本目录不复制原图.
导出、ONNX 与 NPU 比较必须使用同一图片,脚本记录 SHA-256 并核对预处理张量.
校准图片使用 89 已有 COCO 数据,不自动下载.
