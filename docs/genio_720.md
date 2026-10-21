# Genio 720 板端规范

## 平台配置

板端连接、资源目录及工具链参数以 [genio_720.yaml](../configs/platforms/genio_720.yaml) 为准.目标为 Genio 720 EVK,MT8189 / MT8391,MDLA 5.3.系统和 Runtime 版本随每次评价记录,不能由历史文档推断当前设备状态.

运行前确认 Neuron Runtime、APUSYS 设备节点和所需动态库可用.LLM 生成使用系统 `llm_cmdline_tool`,NPU 评价使用匹配的 AArch64 桥接库.组件存在或版本查询成功不代表模型已通过推理和精度验证.

## 编译与部署

编译在 Ubuntu Docker 容器执行,板端执行真实 NPU 推理.使用 `--arch=mdla5.3` 和 `--disallow-bridge`,整数原生 I/O 的转换参数须与实际张量契约一致.系统与模型所需 Runtime 不匹配时停止运行并核对版本,不覆盖系统库强行兼容.

主机入口完成模型、板端程序、配置和必要公共模块的打包上传.在部署目录运行:

```bash
bash run.sh
```

`bash run.sh` 无参数默认固定小批量评价,不改变校准或量化配方.小批量结果使用 `_test` 运行后缀,MiniCPM 沿用 `latest_test` 和 `summary_test.json`;这些是结果标识,不是命令参数.小批量不能覆盖正式全量指标.

## 独立全量精度

在已有编译容器内,从所选模型目录执行 `bash deploy/evaluate.sh`.该入口只对已有浮点模型计算全量参考、准备必要评价输入,不导出、校准、量化、编译、构建、打包或上传.缺少模型时停止,不自动重新部署.

普通模型生成模型资源 `build/onnx_reference.json`,单独放入现有板端 `models/onnx_reference.json`;YAMNet 的参考为模型资源 `fp32_reference.json`,放入板端 `models/fp32_reference.json`.MiniCPM 生成模型资源 `build/native/evaluation/full/`,整体放入板端 `evaluation/full/`,必须与现有 DLA 的上下文、词表和输入契约匹配.这些操作只补充评价资源,不替换 DLA 或运行程序.

在现有板端部署目录执行:

```bash
bash evaluate.sh
```

板端全量入口只推理已有模型、计算精度,不调用 `run.sh`,不运行 Demo、效果渲染、模型大小统计或性能专项.自然产生的诊断耗时可保留在原始记录中.每次结果使用独立的 `<运行编号>_full` 目录,保存预测、原始指标及 `accuracy_summary.json`,不清理预测或覆盖小批量结果.有匹配参考时计算差值,否则注明参考不可用;YAMNet/LLM 缺必要协议输入时直接报错.

MiniCPM 不依赖先运行生成 Demo;完整 MMLU 为 14042 题,WikiText2 使用完整 test 协议.若现有上下文容纳不了完整题目,报告阻塞,另行安排部署调整,禁止截题或在评价中重新编译.DeepSeek 当前只有生成样例,其全量入口明确报告正式协议尚未实现.

## 目录

```text
<board_root>/
├── datasets/                   板端共享评价数据
├── MTK_G720_DLA/               官方 Model Zoo
└── open_models/<model>/        本仓库适配模型
    ├── run.sh                  默认小批量验证
    ├── evaluate.sh             独立全量精度
    ├── config.json             配置快照
    ├── models/                 模型及运行参数
    ├── board/                  板端程序和必要依赖
    └── results/<run_id>/        本次结果、日志与效果示例
```

原生 LLM 的双 DLA、tokenizer 和入口布局由模型 README 定义.正式资源不散放在工作根目录或临时目录.每次评价使用独立运行编号,不覆盖已有结果;淘汰实验不长期保留完整模型副本.

## 评价口径

- 精度使用相同样本、预处理、后处理和评分规则对比原框架、标准格式及 NPU.报告实际覆盖数量、基线和差值.
- 性能分别记录预热后的纯 NPU 调用和端到端耗时,注明输入形状、批量大小、重复次数及计时范围.LLM 的 Prefill 与 Decode 分别使用 token/s.
- 内存注明采集进程和范围,RSS 与系统可用内存变化分别记录.
- 自编样例、固定子集与全量基准必须区分.更换固件、Runtime 或模型后,旧指标不能直接标为新环境结果.

## 系统维护

刷机使用与板卡存储类型匹配的官方完整镜像,按 MediaTek Genio 工具说明操作.镜像、bootloader、kernel、firmware 和 Runtime 应来自匹配的发布组合.刷写和重启会中断设备任务,执行前确认维护授权.

Windows 刷写使用板卡对应的 USB 下载口和官方驱动,UART 用于串口诊断.解包镜像时应保留 Linux 符号链接语义,缺失文件别名时先修正解包结果,不要修改镜像内容.升级后重新确认网络、SSH、APUSYS 和 Runtime,再执行模型验证.
