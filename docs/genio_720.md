# Genio 720 板端验证规范

## 设备基线

```text
设备: MediaTek Genio 720 EVK
SoC: MT8189 / MT8391
系统: Rity Demo 26.0-release (scarthgap)
内核: Linux 6.6.137-mtk+ga246e0c68c39-g429091ed5965
地址: root@192.168.0.92
测试目录: /root/hailong.he
Neuron Runtime: 8.2.16
ONNX Runtime: 1.20.2
GAI 工具: /usr/sbin/llm_cmdline_tool
```

2026-09-07 已通过设备树确认该板为 `MediaTek Genio 720 EVK`,兼容标识包含
`mediatek,mt8391-evk`、`mediatek,mt8391` 和 `mediatek,mt8189`.MediaTek 官网 G720
硬件规格为 NPU 8.0、1× MDLA 5.3,与 `configs/platforms/genio_720.yaml` 一致.

2026-09-21 已通过 Windows 主机、`USB 3.2 P0` 和 Genio Tools 1.7.1 将官网
2026-07-29 发布的正式 v26.0 eMMC 镜像写入开发板.`genio-flash` 已完成 `mmc0`、
`mmc0boot0` 和 `mmc0boot1` 的擦除、写入与重启.刷写后通过串口和 SSH 重新采集了
系统、内核、Runtime、ORT、APUSYS 和网络状态.

Windows 刷写主机上的镜像和工具位于：

```text
D:\data\MTK\tools\scarthgap_k6.6_v26.0_genio-720-evk_private_260729015554.tar.gz
D:\data\MTK\tools\scarthgap_k6.6_v26.0_genio-720-evk_private_260729015554\genio-720-evk
D:\data\MTK\tools\genio-venv
D:\data\MTK\tools\android\platform-tools
```

`neuronrt -v` 当前输出 `Version: 8.2.16`.`onnxruntime` 1.20.2 可以枚举
`NeuronExecutionProvider`、`XnnpackExecutionProvider` 和 `CPUExecutionProvider`.
`llm_cmdline_tool` 由 `neuropilot-bin` 提供,二进制存在且能够进入参数解析；未提供 YAML
配置时会按预期退出.这些检查只证明组件存在和基础加载成功,不等于任何 DLA、ONNX、
LLM 或 VLM 模型已经在正式 v26.0 上完成推理.

APUSYS 当前可见 `/dev/apusys`、`/dev/apuext` 和 `/dev/apusys_apummu`,并加载
`apusys_np8` 与 `apu_top_np8`.这属于 NPU 运行前提证据,不能替代真实模型推理.

完整官网来源、实测命令摘要和对应关系见
[2026-09-07 环境核对记录](genio_720_environment_audit_20260907.md).
本次实际刷写和升级后证据见
[2026-09-21 v26.0 升级记录](genio_720_v26_upgrade_20260921.md).

## 网络与访问基线

板端以 NetworkManager 管理有线网卡 `end0`：

```text
IPv4: 192.168.0.92/24
网关: 192.168.0.2
DNS: 192.168.0.68, 183.221.253.100
SSH: root 空密码,Dropbear socket enabled/active
```

正式镜像默认同时启动的 `dhcpcd.service` 会给 `end0` 追加 DHCP 地址并覆盖默认路由.
当前设备已将该服务设置为 `disabled/inactive`,由 NetworkManager 单独维护静态地址.
Windows 侧已实际执行 `ssh root@192.168.0.92` 并验证 `uid=0(root)`.

## 编译器与运行时兼容边界

主机使用 NCC 8.2.31,板端使用 Neuron Runtime 8.2.16.两者版本不相同,官网没有提供
覆盖全部模型的通用兼容承诺.当前 YOLOv5s 已通过 `--arch=mdla5.3`、`--suppress-output`
和 `--disallow-bridge` 完成针对性验证；该结果不能外推到其他模型.

每个新模型必须至少确认：

1. NCC 目标架构是 `mdla5.3`.
2. 编译计划没有使用 Runtime 8.2.16 或 MT8189 不支持的桥接目标.
3. DLA 在 92 上能够加载并使用 `-m hw` 完成真实输入推理.
4. 日志中没有被忽略的版本、目标或动态库错误.

正式 v26.0 刷入前产生的模型结果仍是有效历史证据,但不能描述为正式 v26.0 的运行结果.
升级后必须重新执行真实模型加载、推理、精度和性能流程,再更新对应模型状态与报告.

## 性能口径

每个模型至少报告：

1. `neuronrt` 预热后的纯 NPU 平均延迟.
2. Demo 预处理、NPU、后处理合计的端到端延迟.
3. 推理进程峰值常驻内存.
4. 重复次数、性能模式、输入形状、数据类型和批量大小.

延迟测试必须使用真实输入并至少进行预热.MTK 官网数据只能作为参考,不能写入"本项目实测".

## 板端目录规范

板端根目录按来源划分模型,测试数据统一进入共享目录：

```text
/root/hailong.he/
├── datasets/                  # 共享评测数据集.
├── wheels/                    # 共享离线 wheel.
├── archive/<date>/            # 历史探针与旧目录,不参与当前运行.
├── MTK_G720_DLA/              # MediaTek 官方 Model Zoo.
└── open_models/<model>/       # 本仓库适配的开源模型.
    ├── models/                # DLA、ONNX 与固定运行辅助文件.
    ├── demo/                  # 少量输入的冒烟与公开示例.
    ├── eval/<run_id>/         # 独立正式评测运行及其证据.
    └── runs/<run_id>/         # 需要独立工作目录的模型运行.
```

禁止在 `/root/hailong.he` 根目录直接放置 DLA、输入 bin、探针输出或模型专属运行目录.
正式数据集保存在 `datasets/`,包括 Whisper-Tiny 板端 Mel 输入。脚本以 `demo/`、
`eval/<run_id>` 或 `runs/<run_id>` 作为模型输出位置；
每个新的正式运行必须使用新的 `run_id`,不得覆盖既有证据.

`tools/board/normalize_genio720_layout.sh` 仅用于旧版目录的一次性整理,不适用于已存在
`open_models/` 或 `MTK_G720_DLA/` 的新布局；历史评测文档保留当时的实际路径.

## 精度口径

同一份验证集、同一份预处理和同一套后处理分别评测：

- 原始 PyTorch 或来源框架模型.
- FP32/FP16 ONNX 模型.
- MTK NPU DLA 模型.

报告中同时给出绝对指标和相对原始模型的精度变化.
