# FastSAM-s

编译与转换步骤统一在 Docker 容器内执行: 先在仓库根目录运行 `bash docker/enter_container.sh`,再进入本模型目录运行原有 `deploy/run.sh` 或 `deploy/run_native.sh` 入口.模型脚本不再调用 Docker 或在主机和容器之间复制文件; 上传完成后,板端入口仍在开发板执行.

类别无关实例分割，输入为 640×640 RGB 图像。来源见[模型卡](model_card.md),资源下载地址见 [source_url.txt](models/source_url.txt)。

量化方式: INT8 训练后量化 (PTQ),采用逐输出通道权重量化.

## 配置与执行

模型和任务参数在模型根目录的 [config.yaml](config.yaml) 中修改,与 README 同级,每项带中文备注.权重和产物的相对路径以模型外部资源目录为基准,数据集根目录由仓库本地存储配置提供; 校准图片与导出检查图片的相对路径以该数据集目录为基准.

所有主机步骤的 `--config` 默认指向这个 YAML,直接运行 Python 文件即可读取; 也可以显式指定其他配置文件,导出和量化的显式参数继续覆盖 YAML 默认值.

SDK、OpenCV 交叉编译库和编译器统一读取仓库的 [公共工具链配置](../../../../configs/toolchains/neuropilot_8_0_11.yaml),不在每个模型 YAML 中重复填写.已有环境变量 `NCC_ROOT`、`MTK_G720_CPP_TOOLCHAIN_ROOT`、`MTK_NEURON_INCLUDE`、`CROSS_CXX` 优先于公共配置.交叉编译使用容器内的编译器、SDK 头文件与已准备的 OpenCV 库,模型脚本不自动安装依赖.

开发板 SSH 地址统一读取 [公共平台配置](../../../../configs/platforms/genio_720.yaml) 的 `board_host`,用于上传和登录提示,不再在模型 YAML 中重复填写.模型的 `board` 分组只保留本模型的部署目录、数据集目录和结果目录.

89 主机与 Docker 须能访问同一个仓库路径、配置文件和主机数据集路径.板端数据集独立配置.权重与数据集需提前离线放置,资源地址见 [source_url.txt](models/source_url.txt).

先在 89 主机仓库根目录执行 `bash docker/enter_container.sh`,进入容器后在模型目录执行:

```bash
bash deploy/run.sh
```

上传完成后,在 92 开发板的部署目录执行:

```bash
bash run.sh
```

[run.sh](deploy/run.sh) 只判断编译阶段或板端阶段,按顺序调用独立文件并显示进度.各 Python 文件直接在当前容器读取配置并执行,不检测容器、不调用 Docker、不重新启动自身; Shell 不解析配置,不内嵌 Python 代码.需要人工执行时直接查看该文件,README 不重复维护逐步命令.

`run.sh` 不接收命令行参数,通过上传后的 `config.json` 自动识别板端阶段.编译主机执行完成后,再登录开发板运行同一脚本.板端保留原来的 `EVAL_RUN_ID` 与 `EVAL_RESUME` 续跑环境变量.板端读取上传时生成的 `config.json` 快照,不读取旧的 `board_paths.conf`; 人工维护的配置仍是 YAML.

主机中间产物默认保存在配置的 `model.output_dir/build/`,默认即模型外部资源目录下的 `build/`,包含导出记录、校准临时文件、缓存、ONNX 评测汇总和交叉编译程序.各步骤复用同一目录,不随日期变化,不自动清理; `--work-dir` 可以覆盖该目录,覆盖时后续步骤须传入同一路径.此前 `/tmp/hailongcodex/` 下的构建文件不会迁移或删除,旧目录中的产物仍可通过 `--work-dir` 显式复用.

外部资源根目录只保留 `FastSAM-s.pt`、`model_fp32.onnx`、`model_int8.dla` 和 `build/`.ONNX 直接生成到最终路径,不再复制一份.量化模型、运行时参数、PyTorch 基线、导出清单、示例清单、配置快照、来源记录和资源说明均直接放入 `build/`; 子目录只保留 `cache/` 与 `tmp/`.完整文件用途见[资源布局说明](models/README.md).显式指定 `--output-dir` 时直接使用指定目录; 如需后续自动流程使用该目录,应同步调整 YAML 的 `model.output_dir`.

ONNX 评测报告先写入 `build/tmp/`,确认完整的 5000 张同协议评测后发布为 `build/onnx_reference.json`.上传首先检查该报告,缺失或无效时立即停止并提示执行评测步骤.主机与容器直接使用共享 NAS 文件,不再用 `docker cp` 在同一路径之间复制.历史目录整理不会补造缺失的精度报告,也不会执行评测.

2026-10-15 已核对并归档重复 ONNX、旧导出基线与过期部署清单.历史文件已从 NAS 移至 89 主机本地磁盘的 `/tmp/hailongcodex/2026-10-15/nas_cleanup/_archive/2026-10-15/fastsam_01a12f77/`,不参与当前流程.归档中的 `index.json` 保留整理时的原路径、哈希及原因; 本次迁移记录保存在 `nas_cleanup/migration.json`.本次只整理文件和清单,不代表重新完成了转换、编译或精度验证.

主机及 Docker 读取 YAML 需要 PyYAML.导出使用 `ultralytics==8.0.111`,转换使用已有 MTK Converter 与 Neuron SDK.全量 ONNX 评测需要 ONNX Runtime、OpenCV、NumPy、tqdm 和 pycocotools; 板端需要 OpenCV、NumPy、pycocotools 及已有 Runtime 动态库.不会自动安装依赖或下载资源.

容器需按[仓库环境准备说明](../../../../README.md#模型与数据存储)注册仓库根目录一次.各步骤按真实目录导入 `tools.storage` 和 `models.navigation.segmentation.fastsam.deploy.board.fastsam_utils`,普通依赖统一在文件顶部声明,不使用包名映射、源码链接或 `sys.path.insert`.直接编辑原始源码即可; `run.sh` 不负责安装环境.板端仍使用上传的独立程序,不要求安装本仓库包.

## 数据与结果

主机和板端使用同一份 COCO val2017 样本与标注,数据目录包含 `images/` 和 `annotations/instances_val2017.json`,共 5000 张图片.校准默认按文件名排序选取 100 张,数量可在 YAML 中修改.

固定评测协议见 [accuracy_protocol.json](deploy/board/accuracy_protocol.json).详细报告记录类别无关分割 AP 与 AR,最终核心指标为 `segm AR@100`.精度变化以百分点表示,正数为改善,负数为下降.

板端结果保存在配置的结果目录下,每次运行使用独立编号.单图检查点在评测前保留; 汇总成功后清理本次 `work`,保留核心汇总、详细报告和已生成的效果示例.已有汇总不代表本次流程修改已完成运行验证.

## 当前测试结果

以下为仓库已有的实测结果,不代表本次配置与流程重构已完成运行验证.

COCO val2017 全量 5000 张,采用类别无关 segm AR@100.结果见 [summary.json](results/summary.json).

| 指标 | 结果 |
| --- | ---: |
| ONNX FP32 segm AR@100 | 39.02% |
| 板端 NPU segm AR@100 | **37.57%** |
| 精度变化 | **-1.45 个百分点** |
| 板端 NPU 平均耗时 | **14.72 ms** |
| 推理进程峰值 RSS | **64.55 MiB** |

耗时仅统计 `NeuronRuntime_inference` 调用,不包含逐图模型加载及前后处理.峰值 RSS 包含运行库及前后处理,不代表 NPU 专用内存.

## 效果示例

原图叠加板端实例分割区域,展示分数阈值为 0.25.

输入已放在 `examples/input/`,上传时从配置的数据集自动生成来源与样本清单,随部署上传,不纳入 Git.

全量测试自动复用选定样本的板端预测,生成少量效果文件到本次结果目录的 `examples/output/`.
以下三张输出来自本次全量测试的板端预测,保存在 `examples/output/`.

### 示例 1: 室内场景

![室内场景输入](examples/input/sample_1.jpg)

![室内场景分割输出](examples/output/sample_1_segmentation.jpg)

### 示例 2: 熊

![熊输入](examples/input/sample_2.jpg)

![熊分割输出](examples/output/sample_2_segmentation.jpg)

### 示例 3: 滑雪场景

![滑雪场景输入](examples/input/sample_3.jpg)

![滑雪场景分割输出](examples/output/sample_3_segmentation.jpg)

## 板端部署结构

执行 `upload` 后,配置的板端部署目录布局为:

```text
部署目录/
├── run.sh
├── config.json          # 从人工 YAML 生成的配置快照.
├── models/              # 模型与推理所需参数.
├── board/               # 板端程序、评测代码及必要依赖.
├── examples/input/      # 少量示例输入及来源清单.
└── results/             # 汇总及少量效果示例.
```

全量测试结果保存在配置的 `board.results_dir/<运行编号>/summary.json`.汇总成功后保留核心汇总、详细报告和已生成的少量效果示例,失败时保留本次工作目录.将汇总上传为本模型的 `results/summary.json` 后更新 README.
## 文件结构

- `config.yaml`: 模型根目录下人工维护的带备注配置.
- `deploy/run.sh`: 无参数自动入口,直接列出主机六步和板端四步.
- `deploy/host/`: 独立的导出、量化、模型编译、ONNX 评测、C++ 编译与上传脚本.
- `deploy/board/`: 独立的推理、精度评测、示例绘图和汇总脚本,以及 C++ 程序源码.
- `models/`: 原始权重、转换产物与来源说明; 必要的上游源码放在 `models/upstream/`.
- `examples/`: 少量固定输入与实际板端效果输出.
- `results/summary.json`: 上传后的最新测试汇总.

示例图片来源: 示例 1: [原图](http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg) / [图片许可](http://creativecommons.org/licenses/by-nc/2.0/); 示例 2: [原图](http://farm8.staticflickr.com/7434/9138147604_c6225224b8_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/); 示例 3: [原图](http://farm8.staticflickr.com/7015/6795644157_f019453ae7_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/).

## 模型资源路径

原始权重、ONNX、量化模型、DLA 和配套运行参数使用仓库外的模型存储目录.先在 89 编译主机配置 `configs/storage.local.json` 的 `models_root` 和 `dataset_root`,完整内部路径仅保存本地.本模型的外部目录为 `models_root/navigation/segmentation/fastsam`;编译入口自动读取配置,转换产物也写入此目录.仓库中的 `models/` 只保留源码、文档和公开来源记录.板端继续使用上传的部署目录.
