# Genio 720 环境与官方资料核对记录（2026-09-07）

## 核对范围

本记录核对以下四类信息：

1. MediaTek G720 与 NeuroPilot 官方在线文档。
2. Ubuntu 89 编译服务器及其 Docker 镜像、运行容器。
3. Genio 720 EVK（92）系统、SoC 和 Neuron Runtime。
4. 本仓库登记的工具链与平台配置。

初始核对完成后，已按用户授权迁移 Docker 容器，并下载、校验、解包正式 Rity v26.0 镜像。
开发板当前只能通过网络访问，无法通过 USB 接入 89；正式刷写尚未开始，板端系统未改变。

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

容器迁移前 89 上同时存在两个 Docker 镜像；迁移后状态如下：

| 镜像 | 镜像 ID | 状态 |
| --- | --- | --- |
| `hhl_g720_311:ubuntu22.04-np8.0.11` | `006a427a61fd` | 当前容器使用；Ubuntu 22.04.5，已完成运行验证。 |
| `hhl_g720_311:np8.0.11` | `be3e3853ae0e` | 旧 Debian 12 镜像，已删除。 |

迁移后当前运行容器的实测结果：

```text
容器镜像: hhl_g720_311:ubuntu22.04-np8.0.11
容器系统: Ubuntu 22.04.5 LTS
Python: 3.11.11
mtk_converter: 8.16.0
mtk_quantization: 8.2.1
ncc-tflite: 8.2.31
torch: 2.0.0+cu118
onnx: 1.13.1
gcc/g++: 11.4.0
GPU: NVIDIA GeForce RTX 4090 D；Torch CUDA 与 ONNX Runtime CUDA 运算通过
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

### 正式 Rity v26.0 镜像准备

用户已明确同意 MediaTek 软件许可协议并授权下载。官方 eMMC 镜像保存于：

```text
/data/users/hailong.he/data/MTKG720/rity-v26.0-genio-720-evk-emmc/
```

| 项目 | 值 |
| --- | --- |
| 文件 | `scarthgap_k6.6_v26.0_genio-720-evk_private_260729015554.tar.gz` |
| 下载地址 | https://download.mediatek.com/aiot/download/prebuilt/v26.0/scarthgap_k6.6_v26.0_genio-720-evk_private_260729015554.tar.gz |
| 文件大小 | 约 1.9 GB |
| 官网 MD5 | `507f111167fadf707c12d77e0b96e337` |
| 实际校验 | `OK` |
| 解包目录 | `image/genio-720-evk/`，约 4.4 GB |
| Genio Tools | 1.7.1，安装于 `/data/users/hailong.he/data/MTKG720/genio-tools-v1.7.1/` |

压缩包共 125 个成员，未发现绝对路径或 `..` 路径。`genio-flash --dry-run` 已识别为
`Rity Demo Layer 26.0-release`、Scarthgap、`genio-720-evk`，计划擦除并写入
`mmc0`、`mmc0boot0` 和 `mmc0boot1`。镜像内核文件版本为 6.6.137。

板端升级前备份保存于
`/data/users/hailong.he/data/MTKG720/migration_20260907/board_root_hailong.he_before_v26.0.tar.gz`，
并附有 MD5。当前 89 的 USB 枚举尚无 MediaTek `0e8d:0003` 设备，而且开发板无法通过 USB
接入 89，因此实际刷写未开始。

板端只读检查确认 `/dev/mmcblk0p10` 是唯一 `rootfs`，并正挂载为 `/`；启动参数直接使用
`root=PARTLABEL=rootfs`。系统未安装 RAUC、SWUpdate、Mender、OSTree 或 Aktualizr，也没有
OS 的备用根分区。MediaTek 文档中的固件 A/B 分区不覆盖 OS 的 kernel/rootfs，`genio-flash`
daemon 模式仍依赖目标板与刷机主机之间的 USB/fastboot 链路。因此不能把网络可达误认为支持
网络整机刷写，也不能通过 SSH 向正在运行的 eMMC 写入完整 WIC 镜像。

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
| Ubuntu 22.04 当前容器 | 项目验证基线 | 已完成迁移和工具/GPU 运行验证；官网 Neuron SDK 推荐 OS 列表只到 Ubuntu 18.04，因此只能表述为项目实测支持。 |
| Rity 26.0-dev | 同版本线、非正式基线 | 与正式 v26.0 同为 Scarthgap，但板端是较早的开发快照，内核 6.6.117；官网正式 v26.0 为 6.6.137。 |

## 使用约束

1. 当前同名容器已绑定目标镜像；后续重建仍应由 `docker/create_container.sh` 检测镜像 ID，不得静默覆盖容器。
2. 旧容器的 inspect、pip 清单和可写层 diff 仅用于追溯；恢复时应优先从 Git 和显式挂载数据重建。
3. NCC 与 Runtime 必须在每个 benchmark 中分别记录，不能只写“NeuroPilot 8”。
4. 每个新模型必须在 92 上验证 DLA 加载和真实 NPU 推理；YOLOv5s 的成功不能外推为所有模型兼容。
5. 对 MT8189 编译时使用 MDLA 5.3。需要桥接或特殊输出格式的模型应先检查板端 Runtime 8.2.16 的支持情况。
6. 板端升级已获得明确授权；但当前网络连接不能替代官方 USB 下载模式。需要将开发板 USB
   接到任意可运行 Genio Tools 的 Linux/Windows 主机后再刷写，完成后重新采集系统、Runtime
   和模型验证证据。

## 证据边界

### 已实时验证

- 89 宿主系统、镜像清单、迁移后容器关联镜像、容器内工具版本、NCC、Torch CUDA 和 ONNX Runtime CUDA。
- 92 设备树、Rity 版本、内核版本、Neuron Runtime 版本和相关软件包。
- 官网 G720、NeuroPilot 8.0.11、Converter、Neuron SDK、下载页和 IoT Yocto v26.0 页面内容。

### 本次未验证

- 任何模型的重新转换、NCC 编译、DLA 加载、板端推理、精度或性能。
- 正式 Rity v26.0 刷写后的 Neuron Runtime 版本和模型兼容性。
