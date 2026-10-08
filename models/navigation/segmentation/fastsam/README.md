# FastSAM-s

类别无关实例分割，输入为 640×640 RGB 图像。来源见[模型卡](model_card.md),资源下载地址见 [source_url.txt](models/source_url.txt)。

量化方式: INT8 训练后量化 (PTQ),采用逐输出通道权重量化.

## 第一步: 编译并上传

所有配置均在脚本顶部,按模型与校准数据、产物与临时目录、板端地址与数据、ONNX 精度数据、编译环境分组.脚本已填写当前部署环境的路径,使用时按注释调整等号右侧的值; 编译主机和 Docker 须能访问相同数据,板端路径独立配置.

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

## 数据与精度评测

在脚本顶部填写 `ONNX_DATASET_DIR`,必须与板端数据采用同一份样本、标注及评测协议。编译主机在 Docker 中自动评测 ONNX,只记录任务核心精度,不记录主机耗时或内存。

数据目录要求: COCO val2017: images/ 和 annotations/instances_val2017.json,5000 张图片.

精度变化以百分点表示,正数为改善,负数为下降.

Docker 需要 ONNX Runtime、NumPy、OpenCV、tqdm 和 pycocotools。

## 当前测试结果

待上传本模型的 `results/summary.json` 后更新。只记录板端 NPU 平均耗时、推理进程峰值 RSS (MiB)、任务核心精度、同协议 ONNX 参考精度和精度变化。峰值 RSS 包含运行库及前后处理,不代表 NPU 专用内存。

## 效果示例

原图叠加板端实例分割区域,展示分数阈值为 0.25.

输入已放在 `examples/input/`,来源和样本对应关系见 [samples.json](examples/input/samples.json).

全量测试自动复用选定样本的板端预测,生成少量效果文件到本次结果目录的 `examples/output/`.
将这些文件取回本模型的 `examples/output/` 后即可更新效果展示.当前先展示输入,输出以实际板端测试为准.

### 示例 1: 室内场景

![室内场景输入](examples/input/sample_1.jpg)

### 示例 2: 熊

![熊输入](examples/input/sample_2.jpg)

### 示例 3: 滑雪场景

![滑雪场景输入](examples/input/sample_3.jpg)

## 板端部署结构

第一步上传到 `BOARD_DEPLOY_DIR` 后的布局统一为:

```text
部署目录/
├── run.sh
├── board_paths.conf
├── models/              # 模型与推理所需参数.
├── board/               # 板端程序、评测代码及必要依赖.
├── examples/input/      # 少量示例输入及来源清单.
└── results/             # 汇总及少量效果示例.
```

全量测试结果保存在 `BOARD_RESULTS_DIR/<运行编号>/summary.json`.成功后保留汇总和少量效果示例,失败时保留本次工作目录.将汇总上传为本模型的 `results/summary.json` 后更新 README.
## 文件结构

- `deploy/run.sh`: 编译上传和板端测试的唯一 Shell 入口.
- `deploy/host/`: 编译主机使用的导出、转换与辅助工具.
- `deploy/board/`: 板端程序源码、预处理和评测代码.
- `models/`: 原始权重、转换产物与来源说明; 必要的上游源码放在 `models/upstream/`.
- `examples/`: 少量固定输入与实际板端效果输出.
- `results/summary.json`: 上传后的最新测试汇总.
