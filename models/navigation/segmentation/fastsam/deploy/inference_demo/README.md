# FastSAM 板端 Demo

板端程序为 `fastsam_board`,使用 C++ 和 Neuron Runtime API 直接加载 DLA,
通过 OpenCV 完成 JPEG 预处理、掩码还原和叠加图.不依赖板端 Python.
由 89 宿主的 `build_board_cpp.sh` 交叉编译,`deploy_board.sh` 负责单图冒烟.

在部署后的运行目录中执行:

```bash
./fastsam_board --model model_int8.dla --config runtime_config.csv \
  --image input.jpg --output-dir point_demo \
  --point 320 200
```

坐标使用原图像素,输出目录必须不存在.全图模式省略提示参数.
`--point x y` 保留覆盖该前景点的全部实例,不实现负点或多点合并.
`--box x1 y1 x2 y2` 选择与提示矩形掩码 IoU 最大且相交的一个实例.
两个提示互斥,越界坐标报错.这两个入口是轻量 CPU 实例选择,
不等同于完整 FastSAMPrompt 的多提示交互语义.

INT8 原生输出暂按已在本项目其他模型使用的 NCHW 行宽 16 对齐契约解析.
不同图的编译布局不能仅凭字节数确认; 必须审核反量化张量与 ONNX 对照后确认.
C++ 程序在加载 DLA 后检查全部 10 个输出的原生缓冲区大小;
实际输出数值和分割质量仍需与同图 PyTorch/ONNX 结果对照.
使用 `--check-image` 可在尚无模型时验证 C++ 的图片预处理路径,
该模式不会初始化 Neuron Runtime,不能作为 NPU 冒烟结果.
