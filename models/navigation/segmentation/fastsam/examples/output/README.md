# 输出证据

板端证据回收到 `runs/<run_id>/{onnx,npu}/`,每次运行独立保存.
包括原始输出、逐实例 PNG 掩码、overlay.jpg、results.json、比较报告和运行日志.
2026-09-23 C++ 板端冒烟结果位于 `runs/20260923_cpp_fastsam_smoke_v2`;
此目录在 89 和本机保留,不进入普通 Git.
精选叠加图位于 `public/fastsam_s_sample_1_overlay.jpg`,原始图片复用
YOLOv5s 公共样例目录中的 CC0 图片 `000000000001.jpg`.
