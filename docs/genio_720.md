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

支持的模型可使用 `bash run.sh -test` 进行固定小子集评价.快速结果单独标记,不得代替全量报告.具体依赖和数量见模型 README.

## 目录

```text
<board_root>/
├── datasets/                   板端共享评价数据
├── MTK_G720_DLA/               官方 Model Zoo
└── open_models/<model>/        本仓库适配模型
    ├── run.sh                  板端入口
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
