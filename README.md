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
bash models/perception/object_detection/yolov5s/deploy/run.sh
```

主机端完成转换、编译、参考评价和上传.登录开发板,进入入口打印的部署目录执行:

```bash
bash run.sh
```

两端无参数默认固定小批量验证,不改变校准、量化或编译配方.全量精度从部署流程拆出: 容器运行对应模型的 `deploy/evaluate.sh`,只读取已有浮点模型生成全量参考及评价输入;资源单独放入现有板端部署包后,执行 `bash evaluate.sh`,只进行全量推理和精度计算.该入口不导出、量化、编译、构建或上传,也不运行性能专项.结果按运行编号保留预测和 `accuracy_summary.json`,不覆盖已归档指标.各模型支持范围及资源位置见对应 README 和[板端规范](docs/genio_720.md).

新建编译环境时使用 `docker/build_image.sh` 和 `docker/create_container.sh`,具体资源要求见 Docker 文档.运行模型时不自动下载权重、数据集或安装依赖.

## 存储规则

仓库保存代码、配置模板、来源记录、文档和精简结果.所有正式资源位于 NAS:

| 资源 | 位置 |
| --- | --- |
| 官方权重、离线源码、ONNX/TFLite/DLA 和配套参数 | `models_root/场景/类别/模型/` |
| 转换中间产物、构建程序和部署包 | 对应模型资源目录的 `build/` |
| 校准与评价数据集 | `dataset_root/` |
| 一次性临时文件 | `/tmp/hailongcodex/当天日期/`,Windows 使用 `D:/tmp/hailongcodex/当天日期/` |

主机与容器使用相同绝对路径,模型资源读写挂载、数据集只读挂载.正式模型和数据集不得以临时目录为唯一存放位置,也不得依赖指向临时目录的链接.来源记录应包含官方地址、固定版本和文件信息,内部完整路径仅保存在本地配置及运行记录中.

板端使用独立部署包,无需访问源码仓库.每次执行使用独立运行编号,结果保存在模型部署目录的 `results/`.淘汰实验只保留必要结论,不长期堆积完整模型副本.

## 仓库结构

```text
configs/             平台、工具链及本地存储配置模板
models/              按场景和任务组织的模型适配
common/cpp/          共享 C++ Runtime、后处理和构建规则
tools/               公共配置、转换、评价及文件工具
docker/              编译镜像与容器入口
registry/            模型范围、来源索引与状态
reports/             平台报告入口
docs/                环境及开发板使用规范
docs_study/          分阶段学习材料
templates/model/     新模型交付模板
tests/               公共代码回归测试
```

模型入口只编排独立步骤:主机实现在 `deploy/host/`,板端代码在 `deploy/board/`.公共配置和工具集中维护,不在模型脚本重复设置 SDK、Docker 和交叉编译环境.

## 模型与结果

[模型总览](models/README.md)是指标的统一索引.各模型 README 说明使用方式、当前结果和必要的协议限制;`results/summary.json` 保存机器可读汇总.不在首页重复模型指标、实验经过或逐次运行日志.

精度比较须使用同一模型、输入和评价协议.全量、固定子集和功能示例分别标识,运行成功不等于精度合格.性能注明纯 NPU 或端到端范围,RSS 不等同于全部驱动内存.更换权重、量化配方或设备环境后须重新评价才能更新结论.
