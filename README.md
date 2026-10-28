# MTK Models

面向 MediaTek Genio 720 和 Genio 5100 的开源模型适配仓库,包含模型转换、量化、NPU 部署及精度和性能评价.模型必须来自作者或官方开源项目;Qualcomm AI Hub Models 仅作为目录和交付方式参考.

- [MTK 模型部署 skill](.agents/skills/mtk-model-deploy/SKILL.md): 复用本仓库适配与独立评价流程.
- [模型总览](models/README.md): 型号、输入、量化方式和已归档指标.
- [编译环境](docs/environment.md)与 [Docker](docker/README.md): 工具链、依赖和容器配置.
- [Genio 720 板端规范](docs/genio_720.md): 部署布局、运行要求和评价口径.
- [YOLOv5s 学习文档](docs_study/Genio720_MT8189部署YOLOv5s总览.md): 分阶段理解完整部署链路.

## 快速开始

在 `configs/storage.local.json` 配置 NAS 的 `models_root` 和 `dataset_root`,格式见 [配置示例](configs/storage.example.json).将 [G720 平台配置](configs/platforms/genio_720.yaml) 复制为同目录的 `genio_720.local.yaml`,在本地副本中填写开发板地址、容器名称和工具链路径.两份本地配置均不提交 Git;环境变量可覆盖本地平台配置,也可用 `MTK_PLATFORM_CONFIG` 指定本地平台文件.

在 Ubuntu 编译主机的仓库根目录进入已有容器:

```bash
bash docker/enter_container.sh
```

在容器的仓库根目录执行所选模型入口,例如:

```bash
bash models/perception/object_detection/yolov5s/deploy/compile.sh
```

需要顺序处理全部 13 个已有模型时,在容器中执行:

```bash
bash models/deploy/compile.sh
```

总入口先将仓库中的 `models/deploy/compile.sh`、`evaluate.sh` 和 `upload.sh` 上传到本地配置 `board_open_models_root` 下的 `deploy/`,再顺序调用各模型已有的编译入口,完成导出、量化、编译、部署及浮点子集精度评估,各模型保存 `results/summary_sub.json` 并上传同子集参考.这些步骤均包含在本次总入口中,无需另行运行评估命令.登录开发板后,进入该 `deploy/` 目录执行 `bash compile.sh`,顺序运行各模型同一子集的 NPU 推理、任务精度及损失统计与示例生成.所有脚本均在仓库维护,板端只执行上传后的代码.

结果回传使用本地平台配置 `results_host` 指定的编译服务器普通账户、`results_root` 指定的仓库根目录.容器编译入口调用 `tools/storage.py --write-upload-config` 从本地配置生成回传设置,再上传到板端 `deploy/upload.local.sh`,该文件不提交 Git.板端使用专用 SSH 密钥 `~/.ssh/mtk_results_ed25519`,服务器主机密钥需预先写入板端 known_hosts.认证配置仅保存在本地,不写入公开脚本.

板端总 `compile.sh` 全部成功后自动回传各模型 `results/summary_sub.json` 和 `examples/output/`;总 `evaluate.sh` 全部成功后自动回传各模型的 `results/summary.json` 和 `examples/output/`,写入编译服务器仓库对应模型路径.可在板端总入口目录执行 `bash upload.sh` 单独补传,或执行 `bash upload.sh examples` 只传示例.回传不执行推理、不传 DLA 或临时预测、不删除服务器其他文件;缺文件时明确报缺失并返回非零状态,不生成空报告.回传文件不会自动提交 Git.

正式评价分两次执行:容器中运行 `bash models/deploy/evaluate.sh` 准备已有模型的全量参考,按各模型终端提示传输必要评价资源后,在板端部署根目录的 `deploy/` 中运行 `bash evaluate.sh`.总入口显示模型进度,遇到首个失败立即停止并显示模型和退出码;修复后也可直接运行该模型的入口.不跳过失败模型,不另建总报告.全量指完整执行各模型已约定的协议范围,MiniCPM 的 MMLU 仍为固定 1000 题.

ONNX 文件按原始权重中的模型名称命名,去除权重校验后缀,例如 `resnet101-cd907fc2.pth` 导出为 `resnet101.onnx`,`yolov8n.pt` 导出为 `yolov8n.onnx`.兼容图保留 `_mtk_compatible` 后缀,多子图保留模型名和子图名,具体见[模型目录](models/README.md).本次统一命名不自动移动或删除已有资源;旧名称产物需要通过编译入口重新导出,独立评价读取新名称文件.

所有模型的主机端完成转换、编译和上传后结束.请登录开发板,在终端提示的部署目录执行 `bash compile.sh`,完成少量通路检查并生成展示示例.

`compile.sh` 按执行环境区分职责: 主机完成模型转换、量化、编译、打包及上传,不自动启动板端推理;用户在开发板执行同名入口检查通路并保存 README 所需的展示示例.上传完成只标为待板端检查,不代表已验证或已生成示例.主机还准备固定子集浮点精度参考;板端 compile 打印同子集任务精度及损失,保存 results/summary_sub.json,不改写全量 results/summary.json.正式精度、性能及 RSS 报告仅由 evaluate.sh 生成.

入口直接按顺序调用命令,只补充必要阶段提示,工具输出和错误直接显示在终端,不默认保存编译日志.公共环境在创建 Docker 时统一设置,板端路径使用公共工具.

也可以在模型目录单独执行主流程文件,例如 YOLOv5s 在容器中执行 `python deploy/host/compile_model.py`,无需传配置路径.板端使用 `python3 board/infer.py` 做默认小批量推理,`python3 board/render_examples.py` 读取当天同模型、同范围的最近一次结果.独立评价文件默认全量,需要已有全量推理结果;缺少前一步产物时直接报错,不会补跑.完整评价仍运行 `bash evaluate.sh`.

`evaluate.sh` 独立使用已有模型: 容器端生成浮点参考及必要评价输入,放入现有板端部署包后,板端执行 `bash evaluate.sh`,统一给出精度、性能和推理进程峰值 RSS,不重新量化或编译.ResNet101 的容器入口已自动上传参考,具体行为见模型说明.每个模型的 `results/` 保留全量 `summary.json` 和独立子集 `summary_sub.json`,示例位于 `examples/`,板端预测、日志及中间报告位于本模型部署目录的 work/,每次推理覆盖.后续新增模型遵循同一方式,详见[板端规范](docs/genio_720.md).

新建编译环境时使用 `docker/build_image.sh` 和 `docker/create_container.sh`,具体资源要求见 Docker 文档.运行模型时不自动下载权重、数据集或安装依赖.

## 存储规则

仓库保存代码、配置模板、来源记录、文档和精简结果.所有正式资源位于 NAS:

| 资源 | 位置 |
| --- | --- |
| 原始权重、离线源码、下载包、许可证和来源记录 | 对应模型资源目录的 `original/` |
| 当前导出 ONNX、最终 DLA | 对应模型资源目录最外层 |
| TFLite、校准数据和转换编译中间产物 | 对应模型资源目录的 `build/` |
| 多文件模型的完整部署包 | 对应模型资源目录的 `models/` |
| 主机评价、数值对照、缓存与临时文件 | 主机本地日期临时目录,不写 NAS |
| 校准与评价数据集 | `dataset_root/` |
| 一次性临时文件 | `/tmp/hailongcodex/当天日期/`,Windows 使用 `D:/tmp/hailongcodex/当天日期/` |

主机与容器使用相同绝对路径,模型资源读写挂载、数据集只读挂载.正式模型和数据集不得以临时目录为唯一存放位置,也不得依赖指向临时目录的链接.来源记录应包含官方地址、固定版本和文件信息,内部完整路径仅保存在本地配置及运行记录中.

板端使用独立部署包,无需访问源码仓库.所有模型的过程文件统一保存在各自部署目录的 `work/`,compile 和 evaluate 开始推理时覆盖上次中间结果,后续评分和展示复用同一目录.板端不按日期或运行编号建立目录,也不写入 /tmp 内存盘;运行编号仅作为报告内部记录.最终评价报告保存在 `results/summary.json`,示例保存在 `examples/output/`.编译服务器的一次性诊断仍使用日期临时目录.

## 仓库结构

```text
configs/             平台、工具链及本地存储配置模板
models/              按场景和任务组织的模型适配
tools/               公共配置、转换、Python Runtime、评价及文件工具
docker/              编译镜像与容器入口
registry/            模型范围、来源索引与状态
reports/             平台报告入口
docs/                环境及开发板使用规范
docs_study/          分阶段学习材料
templates/model/     新模型交付模板
tests/               公共代码回归测试
```

模型入口只编排独立步骤:主机实现在 `deploy/host/`,板端代码在 `deploy/board/`.公共配置和工具集中维护,不在模型脚本重复设置 SDK 或 Docker 环境.板端推理、前后处理与评价统一使用 Python,直接调用 MTK SDK 动态库,不构建自有 C++ 推理程序.

## 模型与结果

[模型总览](models/README.md)是指标的统一索引.各模型 README 说明使用方式、当前结果和必要的协议限制;`results/summary.json` 保存机器可读汇总.不在首页重复模型指标、实验经过或逐次运行日志.

精度比较须使用同一模型、输入和评价协议.全量、固定子集和功能示例分别标识,运行成功不等于精度合格.性能注明纯 NPU 或端到端范围,RSS 不等同于全部驱动内存.更换权重、量化配方或设备环境后须重新评价才能更新结论.

原有 11 个已适配模型的 Python 推理代码已迁移,尚未进行板端验证.这些模型文档中的历史精度、性能、RSS 和示例不能作为新实现的验证结果.[ResNet101](models/perception/image_classification/resnet101/README.md)已完成 G720 Python NPU 全量精度、性能与 RSS 评价,同协议 ONNX 参考已核对,当前结果见模型总览和报告.
