# RTMPose Body2d

人体 WholeBody 姿态估计，输入为 256×192 RGB 人体裁剪图，输出 133 个关节点。正式上游为 OpenMMLab MMPose v1.3.2；来源见[模型卡](model_card.md)。

量化方式: INT8 训练后量化 (PTQ),采用逐输出通道权重量化.

## 第一步: 编译并上传

所有配置均在脚本顶部,按模型与校准数据、产物与临时目录、板端地址与数据、ONNX 精度数据、编译环境分组.脚本已填写当前部署环境的路径,使用时按注释调整等号右侧的值; 编译主机和 Docker 须能访问相同数据,板端路径独立配置.

在 [run.sh](deploy/run.sh) 中配置 ONNX、校准图片及标注、模型输出目录、COCO-WholeBody 数据集、Docker 与交叉编译环境、板端地址和部署目录。

在编译主机的本模型目录运行:

```bash
bash deploy/run.sh
```

脚本在 Docker 中量化和编译 DLA，在主机交叉编译板端 C++ 程序并上传。

## 第二步: 开发板测试

登录所配置的开发板，执行脚本打印的 `bash .../run.sh` 命令。板端处理 5000 张图片和 104125 个人体框，结果写入 `BOARD_RESULTS_DIR`。

进入实际配置的部署目录后也可运行:

```bash
bash run.sh
```

## 数据与精度评测

在脚本顶部填写 `ONNX_DATASET_DIR`,必须与板端数据采用同一份样本、标注及评测协议。编译主机在 Docker 中自动评测 ONNX,只记录任务核心精度,不记录主机耗时或内存。

数据目录要求: COCO WholeBody: images/、annotations/coco_wholebody_val_v1.0.json 和 person_detection_results/COCO_val2017_detections_AP_H_56_person.json.

人体检测文件使用官方标准 JSON 数组.板端 C++ 清单工具在内存中包装后读取,无需修改数据文件或增加依赖; 检测框编号和排序保持不变.更新该工具后,需在编译主机重新运行 `bash deploy/run.sh`,编译并上传新的板端程序.

精度变化以百分点表示,正数为改善,负数为下降.

Docker 需要 ONNX Runtime、NumPy、OpenCV、tqdm 和 xtcocotools。
板端需要 NumPy、OpenCV 和 xtcocotools,用于可视化和计算 WholeBody AP.

## 当前测试结果

COCO-WholeBody val2017 全量 5000 张图片、104125 个人体框,采用 WholeBody AP.结果见 [summary.json](results/summary.json).

| 指标 | 结果 |
| --- | ---: |
| ONNX FP32 WholeBody AP | 57.04% |
| 板端 NPU WholeBody AP | **53.24%** |
| 精度变化 | **-3.80 个百分点** |
| 板端 NPU 平均耗时 | **3.86 ms** |
| 推理进程峰值 RSS | **32.74 MiB** |

耗时为常驻模型下每个人体裁剪的 `NeuronRuntime_inference` 调用,不包含前后处理,也不是每张完整图片的耗时.峰值 RSS 包含运行库及前后处理,不代表 NPU 专用内存.

## 效果示例

原图叠加板端 WholeBody 关节点和人体、手部骨架.

输入已放在 `examples/input/`,编译时从配置的数据集自动生成来源与样本清单,随部署上传,不纳入 Git.

全量测试自动复用选定样本的板端预测,生成少量效果文件到本次结果目录的 `examples/output/`.
将这些文件取回本模型的 `examples/output/` 后即可更新效果展示.当前先展示输入,输出以实际板端测试为准.

### 示例 1: 骑车姿态

![骑车姿态输入](examples/input/sample_1.jpg)

### 示例 2: 站立姿态

![站立姿态输入](examples/input/sample_2.jpg)

### 示例 3: 街头多人

![街头多人输入](examples/input/sample_3.jpg)

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

示例图片来源: 示例 1: [原图](http://farm7.staticflickr.com/6199/6207968507_effbfb5a1f_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/); 示例 2: [原图](http://farm6.staticflickr.com/5328/8789640387_14e10562cb_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/); 示例 3: [原图](http://farm4.staticflickr.com/3446/3232237447_13d84bd0a1_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/).
