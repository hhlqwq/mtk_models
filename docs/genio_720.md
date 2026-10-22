# Genio 720 板端规范

## 平台配置

板端连接、资源目录及工具链参数以 [genio_720.yaml](../configs/platforms/genio_720.yaml) 为准.目标为 Genio 720 EVK,MT8189 / MT8391,MDLA 5.3.系统和 Runtime 版本随每次评价记录,不能由历史文档推断当前设备状态.

运行前确认 Neuron Runtime、APUSYS 设备节点和所需动态库可用.LLM 生成使用系统 `llm_cmdline_tool`,NPU 评价使用匹配的 AArch64 桥接库.组件存在或版本查询成功不代表模型已通过推理和精度验证.

## 编译与部署

编译在 Ubuntu Docker 容器执行,板端执行真实 NPU 推理.使用 `--arch=mdla5.3` 和 `--disallow-bridge`,整数原生 I/O 的转换参数须与实际张量契约一致.系统与模型所需 Runtime 不匹配时停止运行并核对版本,不覆盖系统库强行兼容.

主机入口 `deploy/compile.sh` 完成导出、量化、DLA 编译、板端程序构建和上传,随后自动在部署目录执行 `compile.sh`,用固定少量输入检查硬件推理并生成 README 示例.只打印步骤与成功/失败信息,不计算或生成精度、性能和 RSS 报告.不会进入全量评价.

入口直接顺序调用命令,输出和错误直接显示在终端,只补充必要阶段提示,不默认保存编译日志.Docker 公共环境由创建容器入口统一设置,板端路径复用公共工具.只保存运行、评分和排错必需的数据.

## 独立评价

在已有编译容器的模型目录执行 `bash deploy/evaluate.sh`,只读取已有浮点模型,生成同协议参考与必要输入.不导出、量化、编译或构建.普通参考为模型资源 `build/onnx_reference.json`,放入板端 `models/onnx_reference.json`;YAMNet 参考为模型资源 `fp32_reference.json`,放入板端 `models/fp32_reference.json`.MiniCPM 和 DeepSeek 的模型资源 `build/native/evaluation/full/` 整体放入板端 `evaluation/full/`.这些是评价输入,不属于对外交付报告.

在现有板端部署目录执行:

```bash
bash evaluate.sh
```

评价入口使用已有 DLA 和程序,执行完整精度评价、必要预热和性能统计,采集推理进程峰值 RSS,最终只给出 `results/summary.json`.不调用 compile.sh,不重新量化或编译.重复评价成功后更新这一个报告,运行编号、数据集范围、版本和缺失项写在报告内部.

中间预测、日志、逐任务指标和调试记录位于 `/tmp/hailongcodex/当天日期/<模型>_<阶段>_<运行编号>/`.失败时保留现场,不拿上次指标冒充本次结果,不为失败运行伪造成功报告.成功发布后 results 内只保留 summary.json.临时目录由使用者按需清理.

MiniCPM 和 DeepSeek 要求完整 MMLU 14042 题与 WikiText2 test.若已有上下文不足,明确报错,不截题或自动重编译.原生生成使用官方 Token 输入,样例速度不能标为标准定长基准.新接入评价代码不代表已完成全量实测,成绩以对应版本的实际报告为准.

## 目录

```text
<board_root>/open_models/<model>/
├── compile.sh                 少量通路检查和 README 示例
├── evaluate.sh                正式评价,不重新编译
├── config.json                配置快照
├── models/                    DLA 和必要运行参数
├── board/                     板端程序和公共模块
├── examples/input/            少量展示输入
├── examples/output/           README 使用的结果图或文本
└── results/summary.json        唯一正式报告
```

原生 LLM 双 DLA、tokenizer 与脚本布局按模型说明维护,同样遵守两个入口和唯一 summary.json.正式模型、数据集和评价输入存放在配置指定的资源位置,临时目录不作为正式模型的唯一存储.

## 评价口径

- 精度使用相同样本、预处理、后处理和评分规则对比原框架、标准格式及 NPU.报告实际覆盖数量、基线和差值.
- 性能分别记录预热后的纯 NPU 调用和端到端耗时,注明输入形状、批量大小、重复次数及计时范围.LLM 的 Prefill 与 Decode 分别使用 token/s.
- 内存注明采集进程和范围,RSS 与系统可用内存变化分别记录.
- 自编样例、固定子集与全量基准必须区分.更换固件、Runtime 或模型后,旧指标不能直接标为新环境结果.

## 系统维护

刷机使用与板卡存储类型匹配的官方完整镜像,按 MediaTek Genio 工具说明操作.镜像、bootloader、kernel、firmware 和 Runtime 应来自匹配的发布组合.刷写和重启会中断设备任务,执行前确认维护授权.

Windows 刷写使用板卡对应的 USB 下载口和官方驱动,UART 用于串口诊断.解包镜像时应保留 Linux 符号链接语义,缺失文件别名时先修正解包结果,不要修改镜像内容.升级后重新确认网络、SSH、APUSYS 和 Runtime,再执行模型验证.
