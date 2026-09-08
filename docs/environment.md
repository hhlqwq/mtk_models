# 环境说明

2026-09-07 的官网资料与实际环境核对结果见
[Genio 720 环境与官方资料核对记录](genio_720_environment_audit_20260907.md)。
以下版本表是当前已迁移并验证的项目工具链。

## 版本基线

服务器 `/data/users/hailong.he/data/MTKG720` 已包含以下官方工具：

| 工具 | 版本 | 用途 |
| --- | --- | --- |
| NeuroPilot SDK Basic | 8.0.11 build 20260211 | 主机端工具集合 |
| mtk_converter | 8.16.0 cp311 | PyTorch/ONNX 到 TFLite 转换和 PTQ |
| mtk_quantization | 8.2.1 | 量化工具 |
| ncc-tflite | Neuron 8.2.31 | TFLite 到 DLA 编译 |
| neuronrt | 板端 8.2.16 | DLA NPU 推理 |

主机编译器与板端运行时版本不同。每份 benchmark 都必须同时记录二者，避免只写模糊的
“NeuroPilot SDK 8”。

官网 G720 当前链接到 NeuroPilot 8.0.11 basic，89 上 SDK 包目录与官网下载包
`neuropilot-sdk-basic-8.0.11-build20260211.tar.gz` 完全对应。官网没有声明 NCC 8.2.31
生成的全部 DLA 均兼容板端 Runtime 8.2.16，因此该组合只能按模型实测确认。

## 当前部署状态（2026-09-07）

```text
89 宿主: Ubuntu 24.04.4 LTS / Linux 6.8.0-136-generic
当前镜像: hhl_g720_311:ubuntu22.04-np8.0.11 (006a427a61fd)
当前容器: hhl_g720_311
当前容器系统: Ubuntu 22.04.5 LTS
当前容器 GCC/G++: 11.4.0
```

当前容器中的 Python 3.11.11、Converter 8.16.0、Quantization 8.2.1、NCC 8.2.31、
Torch 2.0.0+cu118 和 ONNX 1.13.1 已通过版本查询确认。Torch CUDA 和 ONNX Runtime
CUDA 已在 RTX 4090 D 上执行运算验证。旧 Debian 12 容器和镜像已删除。

迁移前的容器检查、软件清单和可写层差异已保存到
`/data/users/hailong.he/data/MTKG720/migration_20260907/`，用于追溯，不作为可直接恢复的容器镜像。

## Python 选择

项目使用 Python 3.11。依据是当前 SDK 自带 CPython 3.11 wheel，且官网 Converter 支持
Python 3.5～3.11。旧 YOLOv5 文档中的 Python 3.7 是历史转换链，不作为本项目通用环境。

ONNX 1.13.1 位于官网要求的 `>=1.3,<1.14` 范围，并属于官网列出的充分测试版本。
Torch 2.0.0 位于普通 PyTorch Converter 的 `>=1.3,<2.6` 范围，但不在官网列出的充分测试
版本中，也不满足 PyTorch V2 Converter 的 `>=2.1` 要求。使用 PyTorch V2 Converter 的模型
必须先调整独立环境并重新验证，不能直接沿用当前 Torch 2.0 基线。

## Docker 网络与镜像

当前镜像为 `hhl_g720_311:ubuntu22.04-np8.0.11`。官网 Neuron SDK 页面推荐的主机系统只列出
Ubuntu 14.04/16.04/18.04，因此 Ubuntu 22.04 应描述为“项目验证基线”，不能描述为
“官网明确支持”。Ubuntu 22.04.5 容器已经完成工具版本、NCC、Torch CUDA 和 ONNX Runtime
CUDA 运行验证。

89 服务器构建 Docker 镜像时使用 host 网络。基础镜像内部的 Ubuntu 官方软件源和容器全局
`/etc/pip.conf` 固定为阿里镜像；模型权重、数据集和 MTK 补丁不允许在服务器或板端下载。

NCC 8.2.31 需要 SDK 自带的 `libc++.so.1`。容器只在 `/usr/local/lib` 建立该文件的软链接，
不把整个 SDK `host/lib` 注册到动态链接器，避免 SDK 的 `libstdc++.so.6` 污染 ONNX Runtime。

## GPU 使用边界

89 的 Docker 容器通过 `--gpus all` 使用 NVIDIA GPU。GPU 用于 PyTorch/ONNX 导出及基线
精度评测；MTK Converter 8.16.0 的 INT8 PTQ 和 NCC 编译没有 CUDA 执行接口，仍使用 CPU。
最终部署性能只统计 92 的 Genio 720 NPU，不能用 89 的 GPU 性能替代板端结果。

Torch 2.0 CUDA 11.8 wheel 内的 NVRTC 动态库采用哈希文件名。模型转换脚本会在 wheel 自带
库目录建立 `libnvrtc.so` 标准名软链接，并执行一次 CUDA `Conv2d`，防止只检查设备可见但
cuDNN 实际不可运行。

## Genio 720 C++ 交叉编译补充

板端正式模型评测由 89 的 `/usr/bin/aarch64-linux-gnu-g++` 交叉编译。OpenCV 头文件固定为
官方 4.9.0 源码，下载地址为
`https://github.com/opencv/opencv/archive/refs/tags/4.9.0.tar.gz`，SHA-256 为
`ddf76f9dffd322c7c3cb1f721d0887f62d747b82059342213138dc190f28bc6c`。交叉链接所用
OpenCV 4.9 和 `libneuronusdk_runtime.mtk.so.8` 来自当前 92 板端，只用于 ABI 对齐，不纳入
Git。工具目录为 `/data/users/hailong.he/data/MTKG720/cpp_toolchain/`。

板端标准 COCO 指标使用 `pycocotools 2.0.10` AArch64 wheel，来源为
`https://pypi.org/project/pycocotools/2.0.10/`，SHA-256 为
`075788c90bfa6a8989d628932854f3e32c25dac3c1bf7c1183cefad29aee16c8`。pycocotools 只计算
最终指标；JPEG 预处理、NPU 推理和检测后处理均由板端 C++ 完成。

## 数据集目录与原则

89 服务器数据集根目录固定为：

```text
/data/users/hailong.he/nas_smb/Datasets/open_source/raw
```

容器内以相同绝对路径只读挂载。模型脚本和报告统一记录真实 NAS 路径，不使用 `/datasets` 别名。

- 数据集以相同绝对路径只读挂载到容器。
- 校准子集与正式精度评测集必须分开记录。
- 不允许将 100 张校准图片的指标标记成 COCO 正式 mAP。
- 不允许用单张分类样例推理结果代替 ImageNet Top-1。
