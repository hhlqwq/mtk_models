# FastSAM-s

类别无关实例分割模型,输入为 640×640 RGB 图像.采用 INT8 训练后量化 (PTQ),使用逐输出通道权重量化,部署目标为 Genio 720.

模型来源见[模型卡](model_card.md),权重与依赖来源见 [source_url.txt](models/source_url.txt).

## 配置与运行

- [config.yaml](config.yaml): 模型、校准数据、输出和板端部署目录.
- [G720 平台配置](../../../../configs/platforms/genio_720.yaml): 工具链和开发板连接信息.
- `configs/storage.local.json`: 模型和数据集的本地存储根目录,准备方式见[仓库说明](../../../../README.md#模型与数据存储).

在项目根目录进入编译容器:

```bash
bash docker/enter_container.sh
```

进入容器后,在本模型目录执行:

```bash
bash deploy/run.sh
```

上传完成后,登录开发板,在配置的部署目录执行:

```bash
bash run.sh
```

两个入口均无需参数.需要逐步执行时,查看 [deploy/run.sh](deploy/run.sh).

C++ 前后处理与芯片推理实现分开,当前仅提供 G720 实现.进入编译容器后可在 `deploy/board/` 执行 `make` 单独构建,程序输出到外部模型资源的 `build/fastsam_board`.G5100 尚无对应 SDK 与开发板验证,当前不支持构建,不能直接复用 G720 的 DLA 或运行库.

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

板端结果位于配置的 `board.results_dir/<运行编号>/`,包含评测汇总和效果示例.仓库中的 `examples/` 保存固定示例,`results/summary.json` 保存已有测试汇总.


模型参数读取根目录的 `config.yaml`,公共工具链环境在进入 Docker 时自动加载.部署入口只编排独立步骤,C++ 使用 `deploy/board/Makefile` 构建.板端沿用原部署目录,快速结果位于 `results/运行编号_test/`.
