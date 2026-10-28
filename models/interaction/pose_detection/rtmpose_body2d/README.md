# RTMPose Body2d

> Python 推理代码已迁移,尚未做板端验证.下文报告和示例属于迁移前的 C++/原生 CLI 实现,不能作为当前 Python 实现的精度、性能或 RSS.

人体 WholeBody 姿态估计，输入为 256×192 RGB 人体裁剪图，输出 133 个关节点。正式上游为 OpenMMLab MMPose v1.3.2；来源见[模型卡](model_card.md)。

量化方式: INT8 训练后量化 (PTQ),采用逐输出通道权重量化.

## 执行范围

先在已有编译容器执行 `bash deploy/compile.sh`,完成转换、量化、编译与上传后结束.随后登录开发板,在部署目录执行 `bash compile.sh`,做少量通路检查并保存 README 示例到 `examples/output/`.该阶段计算并打印子集任务精度,不发布正式报告,不统计性能和峰值 RSS.

正式评价使用独立的 `evaluate.sh`: 容器端只计算已有浮点模型的同协议参考及必要评价输入,资源单独放入已有部署包后,板端运行 `bash evaluate.sh`,统计精度、性能和推理进程峰值 RSS.正式结果只写入 `results/summary.json`,不重新量化或编译.板端预测、日志及内部指标统一位于本模型部署目录的 work/,每次开始推理时覆盖上次中间结果.详见[板端规范](../../../../docs/genio_720.md).

下文已归档结果保留原始模型版本和评价范围,本次入口调整不代表重新实测.

## 第一步: 编译并上传

模型、数据集和部署路径见 [config.yaml](config.yaml),工具链与 NAS 根目录按[仓库说明](../../../../README.md)配置.

在编译主机的本模型目录运行:

```bash
bash deploy/compile.sh
```

入口在当前容器完成量化、DLA 编译和上传。

## 第二步: 独立评价

上传完成后,请登录开发板,在部署目录执行 `bash compile.sh` 完成通路检查和示例生成.正式评价前,按[板端规范](../../../../docs/genio_720.md)准备同协议参考及必要输入,在已部署目录执行:

```bash
bash evaluate.sh
```

精度、性能和峰值 RSS 统一保存到 results/summary.json,板端过程数据位于本模型部署目录的 work/,每次开始推理时覆盖,不重新量化或编译.

## 数据与精度评测

参考评价使用 `config.yaml` 指定的数据集,与板端保持相同样本、标注和预后处理协议.

数据目录要求: COCO WholeBody: images/、annotations/coco_wholebody_val_v1.0.json 和 person_detection_results/COCO_val2017_detections_AP_H_56_person.json.

人体检测文件使用官方标准 JSON 数组.板端 Python 直接读取,无需修改数据文件; 检测框编号和排序保持不变.

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

compile.sh 复用固定少量样本的板端预测,生成展示文件到部署目录的 `examples/output/`,不依赖全量评价.
以下输出来自本次全量测试的板端预测,展示 WholeBody 133 个关节点.示例目录仅保留三组输入和输出图片:

```text
examples/
├── input/
│   ├── sample_1.jpg
│   ├── sample_2.jpg
│   └── sample_3.jpg
└── output/
    ├── sample_1_keypoints.jpg
    ├── sample_2_keypoints.jpg
    └── sample_3_keypoints.jpg
```

### 示例 1: 骑车姿态

![骑车姿态输入](examples/input/sample_1.jpg)

![骑车姿态 WholeBody 输出](examples/output/sample_1_keypoints.jpg)

### 示例 2: 站立姿态

![站立姿态输入](examples/input/sample_2.jpg)

![站立姿态 WholeBody 输出](examples/output/sample_2_keypoints.jpg)

### 示例 3: 街头多人

![街头多人输入](examples/input/sample_3.jpg)

![街头多人 WholeBody 输出](examples/output/sample_3_keypoints.jpg)

## 板端部署结构

第一步上传到 `BOARD_DEPLOY_DIR` 后的布局统一为:

```text
部署目录/
├── compile.sh
├── evaluate.sh          # 独立精度、性能与 RSS 评价.
├── config.json          # 不含连接地址的模型配置.
├── models/              # 模型与推理所需参数.
├── board/               # 板端程序、评测代码及必要依赖.
├── examples/            # 少量输入、来源清单和展示输出.
└── results/summary.json  # 唯一正式评价报告.
```

评价结果只保存在部署目录的 `results/summary.json`.板端预测、日志和中间指标统一保存在本模型部署目录的 `work/`,每次开始推理时覆盖;展示示例保存在 `examples/output/`.取回唯一报告后据实更新 README.
## 文件结构

- `deploy/compile.sh`: 编译部署与展示示例入口;精度、性能和 RSS 使用独立的 `deploy/evaluate.sh`.
- `deploy/host/`: 编译主机使用的导出、转换与辅助工具.
- `deploy/board/`: 板端程序源码、预处理和评测代码.
- `models/`: 原始权重、转换产物与来源说明; 必要的上游源码放在 `models/upstream/`.
- `examples/`: 少量固定输入与实际板端效果输出.
- `results/summary.json`: 上传后的最新测试汇总.

示例图片来源: 示例 1: [原图](http://farm7.staticflickr.com/6199/6207968507_effbfb5a1f_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/); 示例 2: [原图](http://farm6.staticflickr.com/5328/8789640387_14e10562cb_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/); 示例 3: [原图](http://farm4.staticflickr.com/3446/3232237447_13d84bd0a1_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/).

### 输出误差统计

compile.sh 独立对固定少量输入比较浮点与 NPU 输出,在终端显示 CosSim、RMSE、NRMSE 和实际覆盖范围.普通模型最多选择 5 次输入,不减少原有 compile 通路检查样本数;LLM 比较示例首个 Prefill 块的有效位置,BEVFusion 比较单帧的 NPU 子图.这些结果用于数值检查,不能代替任务精度.全量 evaluate.sh 只计算任务精度、性能和 RSS,不保存或比较全量浮点张量.

参考为普通 NPZ 数组文件,保存在编译机器本地日期临时目录并上传板端 board/output_reference.npz,由下次 compile 覆盖,不使用数据库,不写入 NAS.板端中间结果仍使用本模型 work/.NRMSE 定义为 100 × RMSE / RMS(浮点参考),多输出分别统计.详见[输出误差统计](../../../../docs/output_metrics.md).

### compile 子集精度检查

compile.sh 使用固定的 100 张验证图像,复用正式评价协议,在终端打印同子集浮点精度、NPU 精度及变化.子集成绩只用于快速检查,不更新 results/summary.json;过程数据覆盖保存在本模型 work/.CosSim、RMSE、NRMSE 仍只比较固定少量输入.详见[输出误差统计](../../../../docs/output_metrics.md).
