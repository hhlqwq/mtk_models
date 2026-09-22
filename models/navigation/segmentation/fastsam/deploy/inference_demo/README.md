# FastSAM 板端 Demo

依赖板端 Python 3、NumPy、OpenCV; ONNX 基线还需要 ONNX Runtime CPU EP.
NPU 后端通过 `/usr/sbin/neuronrt -m hw` 调用实际硬件,不使用 CPU 回退.

`run_board.py` 与上级目录的 `fastsam_utils.py` 必须保持相对目录关系.
在部署后的运行目录中执行:

```bash
python3 deploy/inference_demo/run_board.py \
  --backend npu --model models/model_int8.dla \
  --metadata models/model_int8.json --image input.jpg --output-dir point_demo \
  --point 320 200
```

坐标使用原图像素,输出目录必须不存在.全图模式省略提示参数.
`--point x y` 保留覆盖该前景点的全部实例,不实现负点或多点合并.
`--box x1 y1 x2 y2` 选择与提示矩形掩码 IoU 最大且相交的一个实例.
两个提示互斥,越界坐标报错.这两个入口是轻量 CPU 实例选择,
不等同于完整 FastSAMPrompt 的多提示交互语义.

INT8 原生输出暂按已在本项目其他模型使用的 NCHW 行宽 16 对齐契约解析.
不同图的编译布局不能仅凭字节数确认; 必须审核反量化张量与 ONNX 对照后确认.
脚本保留 `native_layout_verified=false`,不得因成功读出文件便自动标成验证通过.
