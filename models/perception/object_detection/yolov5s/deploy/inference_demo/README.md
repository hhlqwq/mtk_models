# YOLOv5s 板端 Demo

本目录提供完整板端闭环：

1. `prepare_input.py` 读取真实图片,并按 TFLite 量化参数生成 NCHW INT8 输入.
2. `run_board.sh` 在 92 使用 `/usr/sbin/neuronrt` 执行真实推理、10 次预热和 100 次性能测试.
3. `postprocess_outputs.py` 在 89 解量化三个检测头,完成 YOLOv5 解码、NMS 和结果绘制.

实际张量为一个 `[1,3,640,640]` INT8 输入,以及三个
`[1,255,80,80]`、`[1,255,40,40]`、`[1,255,20,20]` INT8 输出.量化比例和零点不写死,
由 `prepare_input.py` 从当前 TFLite 读取并保存到元数据.

性能测试使用 `neuronrt -c 100`,同时保存 Runtime 详细日志和 `/usr/bin/time -v` 峰值 RSS.
该 RSS 是推理进程内存,不等同于 NPU 片上内存.
