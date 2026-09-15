# YOLO-World XL 板端 Demo

`run_board.py` 使用板端预装的 ONNX Runtime 1.20.2。`neuron` 模式设置
`NEURON_FLAG_USE_FP16=1`，并把 CPU EP 保留为不支持算子的回退后端。

模型输出是三个尺度上的 COCO 80 类 logits 与四方向距离，Demo 在 CPU 上完成 Sigmoid、
stride 8/16/32 距离框解码、逐类别 NMS 和原图坐标恢复。

建议通过模型根目录的 `deploy/run_board.sh` 执行，以保存系统、模型和输出哈希证据。
