# FastSAM-s

> Python 推理代码已迁移,尚未做板端验证.下文报告和示例属于迁移前的 C++/原生 CLI 实现,不能作为当前 Python 实现的精度、性能或 RSS.

类别无关实例分割模型,输入为 640×640 RGB 图像.采用 INT8 训练后量化 (PTQ),使用逐输出通道权重量化,部署目标为 Genio 720.

模型来源见[模型卡](model_card.md),权重与依赖来源见 [source_url.txt](models/source_url.txt).

## 执行范围

先在已有编译容器执行 `bash deploy/compile.sh`,完成转换、量化、编译与上传后结束.随后登录开发板,在部署目录执行 `bash compile.sh`,做少量通路检查并保存 README 示例到 `examples/output/`.该阶段计算并打印子集任务精度,不发布正式报告,不统计性能和峰值 RSS.

正式评价使用独立的 `evaluate.sh`: 容器端只计算已有浮点模型的同协议参考及必要评价输入,资源单独放入已有部署包后,板端运行 `bash evaluate.sh`,统计精度、性能和推理进程峰值 RSS.正式结果只写入 `results/summary.json`,不重新量化或编译.板端预测、日志及内部指标统一位于本模型部署目录的 work/,每次开始推理时覆盖上次中间结果.详见[板端规范](../../../../docs/genio_720.md).

下文已归档结果保留原始模型版本和评价范围,本次入口调整不代表重新实测.

## 配置与运行

- [config.yaml](config.yaml): 模型、校准数据、输出和板端部署目录.
- [G720 平台配置](../../../../configs/platforms/genio_720.yaml): 工具链和开发板连接信息.
- `configs/storage.local.json`: 模型和数据集的本地存储根目录,准备方式见[仓库说明](../../../../README.md#存储规则).

在项目根目录进入编译容器:

```bash
bash docker/enter_container.sh
```

进入容器后,在本模型目录执行:

```bash
bash deploy/compile.sh
```

上传完成后,请登录开发板,在部署目录执行 `bash compile.sh` 完成通路检查与示例生成.正式评价时按[板端规范](../../../../docs/genio_720.md)准备参考及必要输入,再登录开发板的部署目录执行:

```bash
bash evaluate.sh
```

两个入口均无需参数.需要逐步执行时,查看 [deploy/compile.sh](deploy/compile.sh).

板端预处理、NPU 调用和掩码后处理统一使用 Python,不再单独构建 C++ 程序.G5100 尚无对应 SDK 与开发板验证,不能直接复用 G720 的 DLA 或运行库.

## 数据与评测

使用 COCO val2017 全量 5000 张图片,目录包含 `images/` 和 `annotations/instances_val2017.json`.INT8 校准默认按文件名排序选取 100 张图片.

评测采用类别无关实例分割协议,核心指标为 `segm AR@100`.ONNX 与板端使用相同协议,精度变化以百分点表示.详细定义见 [accuracy_protocol.json](deploy/board/accuracy_protocol.json).

## 已有实测结果

以下数据来自仓库保存的 [summary.json](results/summary.json),评测范围为 COCO val2017 全量 5000 张.流程调整不代表已重新完成评测.

| 指标 | 结果 |
| --- | ---: |
| ONNX FP32 segm AR@100 | 39.02% |
| 板端 NPU segm AR@100 | **37.57%** |
| 精度变化 | **-1.45 个百分点** |
| 板端 NPU 平均耗时 | **14.72 ms** |
| 推理进程峰值 RSS | **64.55 MiB** |

耗时仅统计 `NeuronRuntime_inference` 调用,不包含逐图模型加载及前后处理.峰值 RSS 包含运行库及前后处理,不代表 NPU 专用内存.

## 效果示例

以下图片叠加了已有全量测试的板端实例分割结果,展示分数阈值为 0.25.

### 示例 1: 室内场景

![室内场景输入](examples/input/sample_1.jpg)

![室内场景分割输出](examples/output/sample_1_segmentation.jpg)

### 示例 2: 熊

![熊输入](examples/input/sample_2.jpg)

![熊分割输出](examples/output/sample_2_segmentation.jpg)

### 示例 3: 滑雪场景

![滑雪场景输入](examples/input/sample_3.jpg)

![滑雪场景分割输出](examples/output/sample_3_segmentation.jpg)

示例图片来源: 示例 1: [原图](http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg) / [图片许可](http://creativecommons.org/licenses/by-nc/2.0/); 示例 2: [原图](http://farm8.staticflickr.com/7434/9138147604_c6225224b8_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/); 示例 3: [原图](http://farm8.staticflickr.com/7015/6795644157_f019453ae7_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/).

## 资源与结果

模型资源保存在 `models_root/navigation/segmentation/fastsam/`,原始权重、ONNX 和 DLA 位于该目录根层,其余产物保存在 `build/`.文件用途见[资源布局说明](models/README.md).

评价结果只保存在部署目录的 `results/summary.json`.板端预测、日志和中间指标统一保存在本模型部署目录的 `work/`,每次开始推理时覆盖;展示示例保存在 `examples/output/`.取回唯一报告后据实更新 README.

### 输出误差统计

compile.sh 独立对固定少量输入比较浮点与 NPU 输出,在终端显示 CosSim、RMSE、NRMSE 和实际覆盖范围.普通模型最多选择 5 次输入,不减少原有 compile 通路检查样本数;LLM 比较示例首个 Prefill 块的有效位置,BEVFusion 比较单帧的 NPU 子图.这些结果用于数值检查,不能代替任务精度.全量 evaluate.sh 只计算任务精度、性能和 RSS,不保存或比较全量浮点张量.

参考为普通 NPZ 数组文件,保存在编译机器本地日期临时目录并上传板端 board/output_reference.npz,由下次 compile 覆盖,不使用数据库,不写入 NAS.板端中间结果仍使用本模型 work/.NRMSE 定义为 100 × RMSE / RMS(浮点参考),多输出分别统计.详见[输出误差统计](../../../../docs/output_metrics.md).

### compile 子集精度检查

compile.sh 使用固定的 100 张验证图像,复用正式评价协议,在终端打印同子集浮点精度、NPU 精度及变化.子集成绩只用于快速检查,不更新 results/summary.json;过程数据覆盖保存在本模型 work/.CosSim、RMSE、NRMSE 仍只比较固定少量输入.详见[输出误差统计](../../../../docs/output_metrics.md).
