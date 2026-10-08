# Depth Anything V2 Small

单目相对深度估计，输入为 518×518 RGB 图像。输出只表达单张图内的相对结构，不能直接用于米制测距。来源见[模型卡](model_card.md),资源下载地址见 [source_url.txt](models/source_url.txt)。

量化方式: INT8 训练后量化 (PTQ),采用逐输出通道权重量化.

## 第一步: 编译并上传

所有配置均在脚本顶部,按模型与校准数据、产物与临时目录、板端地址与数据、ONNX 精度数据、编译环境分组.按注释修改等号右侧的值,编译环境通常无需调整.

在 [run.sh](deploy/run.sh) 中配置原始权重、上游源码、校准图片、DA-2K 数据集、模型输出目录、Docker 与交叉编译环境、板端地址和部署目录。

在编译主机的本模型目录运行:

```bash
bash deploy/run.sh
```

脚本在 Docker 中转换并编译 DLA，在主机交叉编译板端 C++ 程序并上传。

## 第二步: 开发板测试

登录所配置的开发板，执行脚本打印的 `bash .../run.sh` 命令；结果写入 `BOARD_RESULTS_DIR`。

进入实际配置的部署目录后也可运行:

```bash
bash run.sh
```

## 数据与精度评测

在脚本顶部填写 `ONNX_DATASET_DIR`,必须与板端数据采用同一份样本、标注及评测协议。编译主机在 Docker 中自动评测 ONNX,只记录任务核心精度,不记录主机耗时或内存。

数据目录要求: DA-2K: images/ 和 annotations.json,1033 张图片、2068 个点对.

精度变化以百分点表示,正数为改善,负数为下降.

Docker 需要 ONNX Runtime、NumPy、OpenCV 和 tqdm。

## 当前测试结果

待上传本模型的 `results/summary.json` 后更新。只记录板端 NPU 平均耗时、推理进程峰值 RSS (MiB)、任务核心精度、同协议 ONNX 参考精度和精度变化。峰值 RSS 包含运行库及前后处理,不代表 NPU 专用内存。

## 板端部署结构

第一步上传到 `BOARD_DEPLOY_DIR` 后的布局统一为:

```text
部署目录/
├── run.sh
├── board_paths.conf
├── models/              # 模型与推理所需参数.
├── board/               # 板端程序、评测代码及必要依赖.
└── results/             # 测试结果.
```

全量测试结果保存在 `BOARD_RESULTS_DIR/<运行编号>/summary.json`.成功后只保留汇总文件,失败时保留本次工作目录.将汇总上传为本模型的 `results/summary.json` 后更新 README.
## 文件结构

- `deploy/run.sh`: 编译上传和板端测试的唯一 Shell 入口.
- `deploy/host/`: 编译主机使用的导出、转换与辅助工具.
- `deploy/board/`: 板端程序源码、预处理和评测代码.
- `models/`: 原始权重、转换产物与来源说明; 必要的上游源码放在 `models/upstream/`.
- `examples/`: 示例输入与输出,按需保留.
- `results/summary.json`: 上传后的最新测试汇总.
