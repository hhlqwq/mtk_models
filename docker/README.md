# Docker 环境

本目录定义 Genio 720 模型转换环境。容器名中的 `311` 对应 Python 3.11；该版本与服务器
现有 NeuroPilot SDK 8.0.11 提供的 `mtk_converter 8.16.0 cp311` wheel 一致。

容器通过 NVIDIA Container Toolkit 的 `--gpus all` 使用 89 的 GPU。GPU 用于 PyTorch/
ONNX 导出和基线精度评测。MTK Converter 8.16.0 未提供 CUDA 执行选项，INT8 PTQ 与 NCC
编译仍使用 CPU；板端最终推理由 Genio 720 NPU 执行。

## 创建

```bash
ssh ubuntu89
cd /data/users/hailong.he/github/mtk_models
source env.sh
bash ./docker/create_container.sh
```

创建脚本不会停止、删除或重建其他容器。同名容器存在时，只启动并验证现有容器。
镜像构建使用 host 网络，并将 Debian 与 pip 永久配置为阿里镜像，以适配 89 服务器的网络
环境。`/etc/pip.conf` 位于容器可写层，容器重启后仍然生效。

## 挂载

| 主机目录 | 容器目录 | 权限 | 用途 |
| --- | --- | --- | --- |
| `/data/users/hailong.he/github/mtk_models` | `/workspace` | 读写 | 项目代码和产物 |
| `/data/users/hailong.he/data/MTKG720` | `/opt/mtk` | 只读 | NeuroPilot/Neuron SDK |
| `/data/users/hailong.he/nas_smb/Datasets/open_source/raw` | 同一绝对路径 | 只读 | 正式评测数据集 |

数据集使用同路径只读挂载，容器和宿主机命令中的路径保持一致，不使用 `/datasets` 别名。

## 进入

```bash
bash ./docker/enter_container.sh
```

容器使用 host 网络，便于从 89 访问 `192.168.0.92`。SSH 凭据不挂载进容器；板端为
`root` 无密码环境，首次连接仍需显式接受 host key。

初始化脚本只向系统动态链接器暴露 SDK 的 `libc++.so.1`。禁止把整个
`neuron_sdk/host/lib` 加入全局链接器缓存，否则 SDK 自带的旧 `libstdc++.so.6` 会覆盖
Debian 系统库并导致 ONNX Runtime 导入失败。
