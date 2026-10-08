# YOLO-World XL

完整测试成功后只保留一个 `summary.json`。预测、缓存与日志在汇总成功后删除,失败时保留本次 `work/`。ONNX 浮点精度由编译主机自动实测,上传到板端后计算精度变化。

模型产物写入 `MODEL_OUTPUT_DIR`,板端测试结果写入 `BOARD_RESULTS_DIR`。脚本不在代码目录生成 Python 字节码缓存。

开放词汇目标检测，当前模型固化 COCO 80 类文本嵌入。使用 MediaTek 官方 ONNX，经 ONNX Runtime Neuron EP 在板端运行。来源见[模型卡](model_card.md)，精度与性能见[当前结果](README.md#当前测试结果)。

## 运行

1. 在 [run.sh](deploy/run.sh) 中配置官方 ONNX、模型输出目录、COCO val2017 数据集、交叉编译工具链、板端 ONNX Runtime 库、板端地址和部署目录。
   在编译主机的本模型目录执行 `bash deploy/run.sh`。脚本生成纯 NPU ONNX、交叉编译板端 C++ 程序并上传。
2. 登录所配置的开发板，执行脚本打印的 `bash .../run.sh` 命令；结果写入 `BOARD_RESULTS_DIR`。

ONNX 图将固定文本注意力和分类矩阵乘法改写为卷积，将通道 Split 改写为 Slice，并在板端完成纯 Neuron 推理。模型输出三尺度 64 通道 DFL logits，C++ 在 CPU 后处理阶段完成 DFL、框解码和 NMS。三张公开样例的 FP32 与板端检测数量均为 10、13、4，逐框最大分数差 0.00535、最大坐标差 0.156 像素；板端 profiling 无 CPU 模型节点。[六张可视化](examples/output/README.md)保存在原有示例目录。

### 精度评测

在脚本顶部填写 `ONNX_DATASET_DIR`,必须与板端数据采用同一份样本、标注及评测协议。编译主机在 Docker 中自动评测 ONNX,只记录任务核心精度,不记录主机耗时或内存。

数据目录要求: COCO val2017: images/ 和 annotations/instances_val2017.json,5000 张图片.

量化方式: 未做离线量化,浮点 ONNX 由 Neuron EP 在线编译.

精度变化以百分点表示,正数为改善,负数为下降.

Docker 需要 ONNX Runtime、NumPy、OpenCV、tqdm 和 pycocotools。

## 当前测试结果

待上传本模型的 `results/summary.json` 后更新。只记录板端 NPU 平均耗时、推理进程峰值 RSS (MiB)、任务核心精度、同协议 ONNX 参考精度、部署精度和精度变化。峰值 RSS 包含运行库及前后处理,不代表 NPU 专用内存。

当前计时为 ORT session.Run,不作为独立 NPU 耗时。
