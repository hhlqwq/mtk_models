# Genio 720 环境与官方资料核对记录（2026-09-07）

## 核对范围

本记录核对以下四类信息：

1. MediaTek G720 与 NeuroPilot 官方在线文档。
2. Ubuntu 89 编译服务器及其 Docker 镜像、运行容器。
3. Genio 720 EVK（92）系统、SoC 和 Neuron Runtime。
4. 本仓库登记的工具链与平台配置。

本次仅执行只读查询。没有启动新容器、编译模型、运行板端推理或升级开发板系统。

## 官方资料来源

| 资料 | 地址 | 核对结论 |
| --- | --- | --- |
| G720 Hardware Specifications | https://neuropilot.mediatek.com/sphinx/g720/html/l1_hardware_capability/l2_hardware_capability/hardware_capability.html | G720 使用 NPU 8.0、1× MDLA 5.3。 |
| G720 NeuroPilot SDK | https://neuropilot.mediatek.com/sphinx/g720/html/l1_supported_neuropilot_sdk/supported_neuropilot_sdk_external.html | G720 当前链接到 NeuroPilot 8 + GAI for Android V。 |
| NeuroPilot 8.0.11 Documentation | https://neuropilot.mediatek.com/sphinx/neuropilot-8-basic-gai-full/html/ | 当前文档版本为 8.0.11 basic，build 对应 20260211。 |
| Converter Tool Installation Guide | https://neuropilot.mediatek.com/sphinx/neuropilot-8-basic-gai-full/html/l1_getting_started/l2_installation/l3_converter/converter_installation.html | 支持 64-bit Linux、Python 3.5～3.11；ONNX 要求 `>=1.3,<1.14`；普通 PyTorch Converter 要求 `>=1.3,<2.6`，PyTorch V2 Converter 要求 `>=2.1,<2.6`。 |
| Neuron SDK Installation Guide | https://neuropilot.mediatek.com/sphinx/neuropilot-8-basic-gai-full/html/l1_getting_started/l2_installation/l3_neuron_sdk_tools/neuron_sdk_tools.html | 推荐 x86-64 与 Ubuntu 14.04/16.04/18.04。该推荐列表未覆盖本项目目标 Ubuntu 22.04。 |
| NeuroPilot Downloads | https://neuropilot.mediatek.com/sphinx/neuropilot-8-basic-gai-full/html/l1_downloads/downloads_customer.html | All-In-One 包为 `neuropilot-sdk-basic-8.0.11-build20260211.tar.gz`；MT8189 专用 Neuron SDK 下载项为 NeuroPilot 8.0.7、`v1.2517.03 build 20250423`。 |
| NeuroPilot 8.0.11 Release Notes | https://neuropilot.mediatek.com/sphinx/neuropilot-8-basic-gai-full/html/l1_release_notes/release_notes_8.0.11.html | 包含 MDLA 5.x 的 PReLU 推理和 BatchMatMul 编译崩溃修复，并增加 MT8189 的 Neuron Studio 1.16.0。 |
| IoT Yocto v26.0 Release Notes | https://genio.mediatek.com/doc/iot-yocto/latest/sw/yocto/release-notes/iot-yocto-v26.0-release-note.html | 正式 v26.0 支持 MT8391（Genio 720/G720），使用 Scarthgap 与 Linux 6.6.137；G720 EVK 正式镜像发布于 2026-07-29。 |

在线文档中的登录受限页面需要有效的 MediaTek 开发者账号。本文只记录核对结论和页面地址，
不保存账号、Cookie、令牌或下载内容。

## 实际环境快照

### Ubuntu 89 编译服务器

```text
地址: hailong.he@192.168.0.89
宿主系统: Ubuntu 24.04.4 LTS (Noble)
宿主内核: Linux 6.8.0-136-generic x86_64
服务器仓库提交: 7693085
服务器仓库状态: clean
```

89 上同时存在两个 Docker 镜像：

| 镜像 | 镜像 ID | 状态 |
| --- | --- | --- |
| `hhl_g720_311:ubuntu22.04-np8.0.11` | `006a427a61fd` | 已安装；标签声明 Ubuntu 22.04、NP 8.0.11、Python 3.11、CUDA 11.8；本次没有启动验证。 |
| `hhl_g720_311:np8.0.11` | `be3e3853ae0e` | 当前容器 `hhl_g720_311` 实际绑定的旧镜像。 |

当前运行容器的实测结果：

```text
容器镜像: hhl_g720_311:np8.0.11
容器系统: Debian GNU/Linux 12 (bookworm)
glibc: 2.36
Python: 3.11.11
pip: 25.2
mtk_converter: 8.16.0
mtk_quantization: 8.2.1
ncc-tflite: 8.2.31
torch: 2.0.0+cu118
onnx: 1.13.1
cmake: 3.25.0
gcc/g++: 未安装
```

SDK 根目录为：

```text
/data/users/hailong.he/data/MTKG720/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211
```

目录中的 `mtk_converter-8.16.0-cp311-...manylinux2014_x86_64.whl` 和
`mtk_quantization-8.2.1-py3-none-any.whl` 与当前安装版本一致。

### Genio 720 EVK（92）

```text
地址: root@192.168.0.92
设备模型: MediaTek Genio 720 EVK
兼容标识: mediatek,mt8391-evk / mediatek,mt8391 / mediatek,mt8189
架构: aarch64
系统: Rity Demo Layer 26.0-dev (scarthgap)
内核: Linux 6.6.117-mtk+ge93786e7114b-gbdfdf5b7370c
内核构建时间: 2026-05-12
Neuron Runtime: 8.2.16
```

`neuronrt -v` 会先报告缺少 `libcmdl.so`、`libcmdl_ndk.mtk.vndk.so` 和
`libcmdl_ndk.mtk.so`，随后正常输出 `Version: 8.2.16`。该现象当前只证明可选 CMDL 库未找到；
是否影响特定模型必须由该模型的真实板端推理验证，不应仅凭版本查询判断。

板端安装了 `mtk-apusys-driver`、`mtk-apusys-firmware`、`mtk-apusys-middleware`、
`mtk-apusys-tools`、`neuropilot-bin` 和 `packagegroup-rity-mtk-neuropilot`。

## 对应关系与风险判断

| 核对项 | 判断 | 依据与限制 |
| --- | --- | --- |
| Genio 720 / MT8391 / MT8189 | 对应 | 设备树与官网平台列表一致。 |
| NPU 架构 `mdla5.3` | 对应 | 官网 G720 规格为 1× MDLA 5.3；模型必须按 `--arch=mdla5.3` 编译。 |
| NeuroPilot SDK 8.0.11 | 对应 | 89 上 SDK 目录名和官网 All-In-One 包文件名一致。 |
| Converter 8.16.0 + Python 3.11 | 对应 | 使用 SDK 自带 CPython 3.11 wheel，Python 版本位于官网支持范围。 |
| ONNX 1.13.1 | 对应 | 位于官网 `>=1.3,<1.14` 范围，也是官网列出的充分测试版本。 |
| PyTorch 2.0.0 | 条件对应 | 位于普通 PyTorch Converter 允许范围，但不在官网列出的充分测试版本中；不满足 PyTorch V2 Converter 的 `>=2.1` 要求。 |
| NCC 8.2.31 + Runtime 8.2.16 | 条件兼容 | 版本不相同，官网没有提供对所有 DLA 的通用兼容保证。YOLOv5s 已通过特定编译参数完成板端验证，但其他模型仍需逐个实测。 |
| 当前 Debian 12 容器 | 不符合目标部署 | 当前运行容器不是仓库定义的 Ubuntu 22.04 镜像，且缺少 GCC/G++。 |
| Ubuntu 22.04 目标镜像 | 项目验证基线 | 镜像已安装；官网 Neuron SDK 推荐 OS 列表只到 Ubuntu 18.04，因此 Ubuntu 22.04 属于项目实测支持，不应表述为官网明确支持。 |
| Rity 26.0-dev | 同版本线、非正式基线 | 与正式 v26.0 同为 Scarthgap，但板端是较早的开发快照，内核 6.6.117；官网正式 v26.0 为 6.6.137。 |

## 使用约束

1. 当前同名容器绑定旧镜像，执行 `docker/create_container.sh` 时应由脚本检测镜像 ID 不一致并停止；不得静默覆盖旧容器。
2. 在迁移到 Ubuntu 22.04 镜像前，不能把当前运行容器描述为 Ubuntu 22.04 环境。
3. NCC 与 Runtime 必须在每个 benchmark 中分别记录，不能只写“NeuroPilot 8”。
4. 每个新模型必须在 92 上验证 DLA 加载和真实 NPU 推理；YOLOv5s 的成功不能外推为所有模型兼容。
5. 对 MT8189 编译时使用 MDLA 5.3。需要桥接或特殊输出格式的模型应先检查板端 Runtime 8.2.16 的支持情况。
6. 板端升级到正式 Rity v26.0 属于系统变更，必须单独评估并获得明确授权，不能在模型验证过程中顺带升级。

## 证据边界

### 已实时验证

- 89 宿主系统、仓库提交、镜像清单、当前容器关联镜像和容器内已安装工具版本。
- 92 设备树、Rity 版本、内核版本、Neuron Runtime 版本和相关软件包。
- 官网 G720、NeuroPilot 8.0.11、Converter、Neuron SDK、下载页和 IoT Yocto v26.0 页面内容。

### 本次未验证

- `hhl_g720_311:ubuntu22.04-np8.0.11` 新镜像的启动与运行时状态。
- 新镜像中的 Torch CUDA、ONNX Runtime CUDA、Converter、NCC 和系统编译器执行结果。
- 任何模型的重新转换、NCC 编译、DLA 加载、板端推理、精度或性能。
- 正式 Rity v26.0 刷写后的 Neuron Runtime 版本和模型兼容性。
