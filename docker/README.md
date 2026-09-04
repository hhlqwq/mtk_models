# Docker 环境

本目录定义 Genio 720 模型转换环境。容器名中的 `311` 对应 Python 3.11；该版本与服务器
现有 NeuroPilot SDK 8.0.11 提供的 `mtk_converter 8.16.0 cp311` wheel 一致。

## 创建

```bash
ssh ubuntu89
cd /data/users/hailong.he/github/mtk_models
source env.sh
./docker/create_container.sh
```

创建脚本不会停止、删除或重建其他容器。同名容器存在时，只启动并验证现有容器。

## 挂载

| 主机目录 | 容器目录 | 权限 | 用途 |
| --- | --- | --- | --- |
| `/data/users/hailong.he/github/mtk_models` | `/workspace` | 读写 | 项目代码和产物 |
| `/data/users/hailong.he/data/MTKG720` | `/opt/mtk` | 只读 | NeuroPilot/Neuron SDK |
| `/data/users/hailong.he/nas_smb/Datasets/open_source/raw` | 同一绝对路径 | 只读 | 正式评测数据集 |

数据集使用同路径只读挂载，容器和宿主机命令中的路径保持一致，不使用 `/datasets` 别名。

## 进入

```bash
./docker/enter_container.sh
```

容器使用 host 网络，便于从 89 访问 `192.168.0.92`。SSH 凭据不挂载进容器；板端为
`root` 无密码环境，首次连接仍需显式接受 host key。
