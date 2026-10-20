# Docker 编译环境

容器提供 Ubuntu 22.04、Python 3.11.11、Torch 2.0.0+cu118、NeuroPilot SDK 8.0.11、Converter 8.16.0、Quantization 8.2.1 和 NCC 8.2.31,并包含 C++ 交叉编译器、SSH/SCP 和 rsync.

ONNX 使用 1.13.1,满足当前 Converter 的 `>=1.3,<1.14` 约束.ONNX Runtime GPU 1.18.0 对应 CUDA 11.8/cuDNN 8; 库兼容关系见 [ONNX Runtime CUDA 文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

## 配置

- [平台配置模板](../configs/platforms/genio_720.yaml): 公共平台规格.复制为同目录的 `genio_720.local.yaml`,在副本中填写容器名称、SDK 路径、交叉编译器和开发板连接信息;本地副本不提交 Git.
- `configs/storage.local.json`: NAS 模型与数据集根目录,可通过 `container_image` 指定已准备的镜像; 该文件不提交 Git.
- 构建参数 `PYTHON_SOURCE_URL`: Python 源码下载地址,默认值见 Dockerfile.

SDK、模型、数据集和交叉编译库需按各自来源记录提前准备.下载地址和资源哈希应随本地资源清单保存,不将个人目录写入公共文档.

## 构建与进入

需要创建环境时,在项目根目录执行:

```bash
bash docker/build_image.sh
bash docker/create_container.sh
```

已有容器时,在项目根目录进入环境:

```bash
bash docker/enter_container.sh
```

进入后执行对应模型的运行入口或独立步骤.容器终端自动加载平台环境,模型入口不负责进入 Docker.板端任务仍在开发板执行.

## 挂载与维护

项目目录和模型资源目录按相同绝对路径挂载,模型资源可写,数据集和 C++ 工具目录只读.容器默认工作目录为项目根目录,创建脚本配置 Git `safe.directory`.

构建使用本地 SDK 作为命名上下文,不把模型或数据集打入镜像.镜像内的依赖版本记录在 `/opt/mtk-build/installed-requirements.txt`.SDK 镜像按其许可使用,不推送公共镜像仓库.

现有容器的镜像或挂载不匹配时,先保留环境再安排重建,创建脚本不自动删除旧容器.平台和资源环境在进入容器时自动初始化,无需手动执行环境脚本.修改配置后重新进入终端即可.

GPU 是否实际参与任务应以执行提供程序和运行证据为准.最终部署性能以开发板 NPU 实测为准,不能用编译主机的 GPU 性能替代.
