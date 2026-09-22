# 输出证据

板端证据回收到 `runs/<run_id>/{onnx,npu}/`,每次运行独立保存.
包括原始输出、逐实例 PNG 掩码、overlay.jpg、results.json、comparison.json 和运行日志.
该目录不将未验证输出作为公开样例; 审核通过后再精选到 public 并同步报告.
当前没有真实 FastSAM 输出.
