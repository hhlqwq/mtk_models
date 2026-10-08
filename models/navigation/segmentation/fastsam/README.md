# FastSAM-s

完整测试成功后只保留一个 `summary.json`。预测、缓存与日志在汇总成功后删除,失败时保留本次 `work/`。ONNX 浮点精度由编译主机自动实测,上传到板端后计算精度变化。

模型产物写入 `MODEL_OUTPUT_DIR`,板端测试结果写入 `BOARD_RESULTS_DIR`。脚本不在代码目录生成 Python 字节码缓存。

类别无关实例分割，输入为 640×640 RGB 图像。来源见[模型卡](model_card.md)。

## 第一步: 编译并上传

所有配置均在脚本顶部,按模型与校准数据、产物与临时目录、板端地址与数据、ONNX 精度数据、编译环境分组.按注释修改等号右侧的值,编译环境通常无需调整.

在 [run.sh](deploy/run.sh) 中配置原始权重、校准图、示例图、COCO val2017 数据集、模型输出目录、Docker 与交叉编译环境、板端地址和部署目录。

在编译主机的本模型目录运行:

```bash
bash deploy/run.sh
```

脚本在 Docker 中转换并编译 DLA，在主机交叉编译 C++ 程序并上传。

## 第二步: 开发板测试

登录所配置的开发板，执行脚本打印的 `bash .../run.sh` 命令。板端逐图保存检查点并计算 COCO 分割指标；结果写入 `BOARD_RESULTS_DIR`。

进入实际配置的部署目录后也可运行:

```bash
bash run.sh
```

### 精度评测

在脚本顶部填写 `ONNX_DATASET_DIR`,必须与板端数据采用同一份样本、标注及评测协议。编译主机在 Docker 中自动评测 ONNX,只记录任务核心精度,不记录主机耗时或内存。

数据目录要求: COCO val2017: images/ 和 annotations/instances_val2017.json,5000 张图片.

量化方式: INT8 训练后量化 (PTQ),采用逐输出通道权重量化.

精度变化以百分点表示,正数为改善,负数为下降.

Docker 需要 ONNX Runtime、NumPy、OpenCV、tqdm 和 pycocotools。

## 当前测试结果

待上传本模型的 `results/summary.json` 后更新。只记录板端 NPU 平均耗时、推理进程峰值 RSS (MiB)、任务核心精度、同协议 ONNX 参考精度和精度变化。峰值 RSS 包含运行库及前后处理,不代表 NPU 专用内存。

## 文件结构

- `deploy/run.sh`: 编译上传和板端测试的唯一 Shell 入口.
- `deploy/python/`: 模型导出、转换、评测与辅助代码.
- `deploy/cpp/`: 板端 C++ 源码.
- `models/`: 模型产物与来源说明.
- `examples/`: 示例输入与输出,按需保留.
- `results/summary.json`: 上传后的最新测试汇总.
