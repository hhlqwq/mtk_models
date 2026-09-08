# Genio 720 板端验证规范

## 设备基线

```text
设备: MediaTek Genio 720 EVK
SoC: MT8189 / MT8391
系统: Rity Demo 26.0-dev (scarthgap)
内核: Linux 6.6.117-mtk+ge93786e7114b-gbdfdf5b7370c
地址: root@192.168.0.92
测试目录: /root/hailong.he
Neuron Runtime: 8.2.16
```

2026-09-07 已通过设备树确认该板为 `MediaTek Genio 720 EVK`，兼容标识包含
`mediatek,mt8391-evk`、`mediatek,mt8391` 和 `mediatek,mt8189`。MediaTek 官网 G720
硬件规格为 NPU 8.0、1× MDLA 5.3，与 `configs/platforms/genio_720.yaml` 一致。

当前系统属于 Rity v26.0 的 Scarthgap 开发版本，不是官网 2026-07-29 发布的正式 v26.0
镜像。官网正式 v26.0 使用 Linux 6.6.137，当前板端为 6.6.117。升级系统属于独立系统变更，
已获得用户明确授权。正式 eMMC 镜像已在 89 下载、校验、解包并通过 `genio-flash --dry-run`；
开发板当前只能通过网络访问，无法通过 USB 接入 89，因此尚未实际写入。官网 `genio-flash`
要求目标板通过 USB 进入 SoC 下载模式；当前镜像又只有一个正在挂载的 `rootfs`，板端没有
OS OTA/A-B 更新组件，不能从 SSH 会话安全地原地覆盖整块 eMMC。刷写后必须重新核对 Runtime
和全部已交付模型。

正式镜像和工具位于：

```text
/data/users/hailong.he/data/MTKG720/rity-v26.0-genio-720-evk-emmc/
/data/users/hailong.he/data/MTKG720/genio-tools-v1.7.1/
```

`neuronrt -v` 当前会先报告缺少 CMDL 相关动态库，再输出 `Version: 8.2.16`。版本查询成功
不等于模型推理成功；每个 DLA 仍需执行真实板端加载和 NPU 推理验证。

完整官网来源、实测命令摘要和对应关系见
[2026-09-07 环境核对记录](genio_720_environment_audit_20260907.md)。

## 编译器与运行时兼容边界

主机使用 NCC 8.2.31，板端使用 Neuron Runtime 8.2.16。两者版本不相同，官网没有提供
覆盖全部模型的通用兼容承诺。当前 YOLOv5s 已通过 `--arch=mdla5.3`、`--suppress-output`
和 `--disallow-bridge` 完成针对性验证；该结果不能外推到其他模型。

每个新模型必须至少确认：

1. NCC 目标架构是 `mdla5.3`。
2. 编译计划没有使用 Runtime 8.2.16 或 MT8189 不支持的桥接目标。
3. DLA 在 92 上能够加载并使用 `-m hw` 完成真实输入推理。
4. 日志中没有被忽略的版本、目标或动态库错误。

## 性能口径

每个模型至少报告：

1. `neuronrt` 预热后的纯 NPU 平均延迟。
2. Demo 预处理、NPU、后处理合计的端到端延迟。
3. 推理进程峰值常驻内存。
4. 重复次数、性能模式、输入形状、数据类型和批量大小。

延迟测试必须使用真实输入并至少进行预热。MTK 官网数据只能作为参考，不能写入“本项目实测”。

## 精度口径

同一份验证集、同一份预处理和同一套后处理分别评测：

- 原始 PyTorch 或来源框架模型。
- FP32/FP16 ONNX 模型。
- MTK NPU DLA 模型。

报告中同时给出绝对指标和相对原始模型的精度变化。
