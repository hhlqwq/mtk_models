# 部署参考

## 配置和资源

从 skill 位置向上定位仓库根目录,不写死用户机器路径.

- 存储入口: `configs/storage.example.json`、本地 `configs/storage.local.json`、`tools/storage.py` 和 `tools/storage.sh`.
- 平台入口: `configs/platforms/<platform>.yaml`、同目录本地覆盖、`MTK_PLATFORM_CONFIG`.加载顺序以 `docs/environment.md` 和当前实现为准.
- 模型入口: `config.yaml` 路径由公共工具解析,不在模型脚本重复设置 NAS、SDK、容器和板端根目录.
- 正式权重、离线源码、ONNX/TFLite/DLA 位于模型资源目录;校准/评价数据位于数据集目录.临时目录不能是正式资源的唯一存储位置.
- 复用已有 Docker.缺依赖时报告具体版本和用途,不要擅自覆盖共享 Python 环境或下载大文件.

## 普通视觉和音频

先进入编译容器,再执行已选择的模型入口:

```bash
bash docker/enter_container.sh
bash models/perception/object_detection/yolov5s/deploy/compile.sh
```

入口只编排独立步骤:主机实现放 `deploy/host/`,板端 Python 实现放 `deploy/board/`.复用 `tools/deploy.py`、`tools/board.py` 与 `tools/evaluation.py`.推理、前后处理、评价和性能统计使用 Python,通过 Python 调用 MTK SDK 动态库执行 NPU 推理.已有 C++ 程序和自编桥接库逐模型迁移,不作为新增模型的依赖.

Shell 入口直接调用 Python,只补充必要阶段提示,工具输出与错误直接显示在终端,不默认增加日志文件或包装函数.不增加 C++ 推理程序的 make 或交叉编译步骤.公共环境复用 Docker 配置,板端临时路径由 tools.board.paths 计算;只保存后续步骤必需的产物.

主流程 Python 文件无参数时读取所属模型配置.普通容器步骤用 python,专用解释器沿用本地配置,板端用 python3.已有参数仅用于覆盖默认值,不在入口重复传递.

G720 已适配的普通 DLA 配方:

```bash
ncc-tflite --arch=mdla5.3 --suppress-output --disallow-bridge input.tflite -o model.dla
```

此配方不是全部 MTK 芯片或模型的通用命令.浮点、多输入、多输出及 LLM 按模型编译实现处理.核对编译后真实 dtype、layout、stride、scale/zero point 和缓冲区字节数,不能直接用 ONNX 逻辑 shape 解释原生输出.

板端部署包包括 compile.sh、evaluate.sh、配置快照、DLA、必要参数、Python 脚本和公共模块,依赖匹配板端架构的 SDK 动态库.主机部署后自动调用板端 compile.sh,少量推理只检查通路并保存 README 示例,不计算或发布精度、性能及 RSS.校准与量化配方不随评价范围变化.

## 原生 LLM

先读模型 `ADAPTATION.md`、配置和转换实现,再选环境及分片方式:

- 明确上下文、Prefill 批长、Decode 单 Token、KV Cache、mask 和 RoPE 规则.
- tokenizer、chat template、BOS/EOS 和停止 Token 与官方参考一致.文本/Token 输入一致性须有实测记录.
- 双 DLA、外置 embedding、tokenizer、配置及 Python 推理代码共同构成交付,依赖匹配的 SDK 动态库,不能只统计单 DLA.生成循环、KV 转移及采样均使用 Python.
- W4A16 与 INT16 原生 I/O 按实际图记录,重新量化后不能复用旧契约.
- MiniCPM 和 DeepSeek 使用各自的 host/convert.py、compile.py、package.py 及板端入口,共享量化适配、KV 转移和评价工具.不得跨模型复制特殊 Token、聊天模板或 KV Head 配置.
- 编译阶段仅准备生成示例所需 Token 与运行参数,不加载浮点模型计算基准.完整 MMLU/PPL 参考和评价输入在 evaluate.sh 阶段准备.
- Prefill/Decode 测速及 RSS 采集只属于 evaluate.sh,输入长度、生成计数及内存范围统一写入 summary.json.
- 生成文本放 examples/output,日志和中间指标放临时目录,results 只留一份报告.

## 故障定位

| 现象 | 优先核对 | 边界 |
| --- | --- | --- |
| `EDPA_1_2` / unsupported target | 当前 DLA 参数、MDLA 目标及 Runtime | G720 检查禁止桥接配方;重新编译属于部署修复,不从评价自动触发 |
| `libcmdl.so` / 模型加载失败 | ELF 架构、Runtime、APUSYS、驱动和固件 | 不跨 SoC 覆盖系统库或创建假链接;版本查询不代替真实推理 |
| 输出错位 / 精度异常 | shape/dtype/stride、缓冲区大小、scale/zero point 和预后处理 | 用同一输入逐阶段定位,不改评价阈值掩盖问题 |
| 全量题目超上下文 | 完整题目长度和已有 DLA 上下文 | 报告阻塞,另行计划部署变更,不得截题或在评价中重新转换 |
| 缺样本 / 参考不匹配 | 样本 ID、覆盖、权重及协议 | 保留错误和预测,不混用子集与全量指标 |

保留原始错误、失败阶段和最小复现输入.组件不匹配时规划官方匹配发布组合的维护,不能为单模型覆盖系统库.维护前确认设备任务及授权.
