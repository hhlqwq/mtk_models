# 输出证据

板端证据回收到 `runs/<run_id>/{onnx,npu}/`,每次运行独立保存.
包括原始输出、逐实例 PNG 掩码、overlay.jpg、results.json、比较报告和运行日志.
2026-09-23 C++ 板端冒烟结果位于 `runs/20260923_cpp_fastsam_smoke_v2`;
此目录在 编译主机 和本机保留,不进入普通 Git.
`public/` 还包含与 YOLOv5s 一样平铺的三张板端叠加图。
2026-09-30 使用独立 ImageNet 100 张校准的 DLA 在 Genio 720 重新推理并覆盖原有三图，
结果文件为
`sample_1_segmentation.jpg` 至 `sample_3_segmentation.jpg`，以及同名 JSON。
三张输入复用 YOLOv5s 公共样例目录中的 CC0 图片；JSON 内保存分数、框、
掩码 COCO RLE 和逐图耗时。三图仅用于输出展示，不替代完整 COCO AP。
同一三图的 FP32 ONNX 结果保存为 `sample_1_fp32_segmentation.jpg` 至
`sample_3_fp32_segmentation.jpg` 及同名 JSON。两端均使用置信度 `0.4`、NMS IoU `0.9`、
最多 30 个实例。三图各有 30 个输出实例；按掩码 IoU 最优配对，IoU 不低于 `0.5` 的
实例数分别为 28、29、27。尾部低分实例存在差异，不能把三图称为逐实例完全一致。
旧精选叠加图 `public/fastsam_s_sample_1_overlay.jpg` 作为历史冒烟示例保留。
