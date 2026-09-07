# Ubuntu 预装工具链镜像

镜像: `hhl_g720_311:ubuntu22.04-np8.0.11`, 容器: `hhl_g720_311`。

构建时安装 Ubuntu 22.04、Python 3.11.11、CUDA 11.8 Torch 2.0.0、NeuroPilot SDK 8.0.11、
Converter 8.16.0、Quantization 8.2.1、NCC 8.2.31 和 YOLOv5/COCO 依赖。
ONNX Runtime GPU 1.18.0 对齐 CUDA 11.8/cuDNN 8, pip 永久使用阿里镜像。

## 构建与创建

在 89 的指定仓库执行:

```bash
cd /data/users/hailong.he/github/mtk_models
bash docker/build_image.sh
bash docker/create_container.sh
```

构建使用命名上下文读取服务器已有 SDK, 完整复制到镜像 /opt/mtk。运行时不再挂载
Dockerfile 使用 Docker 内置前端, 不额外下载 docker/dockerfile 镜像。
Ubuntu 系统包保留基础镜像的官方软件源, 避免第三方镜像索引不可用导致构建失败。
Python 3.11.11 源码默认从可达的阿里云镜像获取, 可通过 `PYTHON_SOURCE_URL` 构建参数覆盖。
宿主机 SDK, 创建和模型转换均不再安装 pip 包。构建日志和 GPU 校验通过后才视为环境就绪。

仅项目目录映射为 /workspace, 数据集以原绝对路径只读挂载:
`/data/users/hailong.he/nas_smb/Datasets/open_source/raw`。
容器支持 GPU, 启动校验执行 Torch Conv2d 和 ONNX Runtime CUDA 运算。
实际依赖版本保存在镜像 /opt/mtk-build/installed-requirements.txt。

旧同名容器不会被脚本自动删除, 镜像不匹配时明确报错。先构建验证新镜像, 再安排迁移。
SDK 属于本地供应商资料, 镜像仅保存在 89, 不推送公共镜像仓库。
GitHub 仅提交构建代码, 模型和数据集继续由用户下载。

## GPU 证据边界

此前“MTK PTQ 必然只使用 CPU”的结论证据不足: 没有显式 CUDA 参数不能证明内部执行设备。
已观察到 GPU 导出后 PTQ 加快, 具体原因需通过运行时 GPU 进程监测验证。
板端正式性能以 92 的 NPU 结果为准。

ONNX GPU 兼容依据: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
