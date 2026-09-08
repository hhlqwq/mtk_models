# YOLOv5s 最终交付证据

本目录保存 2026-09-08 在 Genio 720 EVK 完成的正式运行摘要，运行 ID 为
`20260908_cpp_delivery_v3`。

- `coco_metrics.json`：5000 张 COCO val2017 的正式指标。
- `timing_summary_current_run.json`：分阶段时延与进程峰值 RSS。
- `run_inputs_manifest.txt`：提交、代码、模型、评测器、标注、图片清单及协议绑定。
- `run_outputs_sha256.txt`：板端完整运行目录内关键输出的 SHA-256。
- `system_before.txt`、`system_after.txt`：运行前后系统及 CPU 调频状态。

逐图预测、逐图时延、完整图片哈希清单和运行日志保留在 92 的
`/root/hailong.he/yolov5s_cpp/runs/20260908_cpp_delivery_v3/`，并回传到 89 的同名结果目录，
不进入普通 Git 历史。板端 RTC 未同步，因此系统快照中的采集时间不可作为运行日期。
