# Docker 编译环境

基于 Ubuntu 22.04,预装 Python 3.11、Torch、NeuroPilot Converter、LLM SDK 和 NCC,用于视觉模型及原生 LLM 的转换与评价.依赖版本由 Dockerfile 和 requirements.txt 固定.

## 配置

平台路径和连接信息放在 `configs/platforms/genio_720.local.yaml`,模型与数据集路径放在 `configs/storage.local.json`,本地配置不提交 Git.
基础 SDK 和 LLM SDK 使用已准备的本地资源;LLM SDK wheel 目录由本地平台配置的 `toolchain.gai_sdk_root` 指定.模型和数据集不打入镜像.

## 创建与使用

在仓库根目录构建镜像并创建容器:

```bash
bash docker/build_image.sh
bash docker/create_container.sh
```

进入已经运行的容器:

```bash
bash docker/enter_container.sh
```

进入后直接使用 `python`、编译器和其他工具,执行模型入口或独立步骤.依赖在镜像构建时安装,项目路径和平台环境在创建容器时设置;进入脚本只打开终端,不安装依赖或切换解释器.

## 维护

项目、模型、数据集和交叉编译资源按本地配置挂载.现有容器不会因修改构建文件而自动更新,更换镜像或容器前需保留必要资源并确认维护安排.
构建过程和最终依赖清单保存在镜像内,SDK 镜像按其许可使用,不推送公共镜像仓库.最终 NPU 性能和精度以开发板实测为准.
