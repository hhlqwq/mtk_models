---
name: mtk-model-deploy
description: 在 mtk_models 仓库适配和部署官方开源模型到 MediaTek Genio,统一以 compile.sh 编译部署和生成示例、evaluate.sh 评价精度性能与 RSS,每模型只交付一个 summary.json.用于新增模型、修复 MTK 部署链路及排查精度差异.
---

# MTK 模型部署

所有已有及后续新增模型都交付明确张量契约、两个主要入口、必要展示示例和唯一评价报告.优先复用当前仓库工具,不要为单个模型另造环境管理、报告格式或流水线框架.

## 开始工作

1. 读取适用的 `AGENTS.md`,检查 `git status --short`,保留用户改动.只读定位模型 README、配置、模型卡、来源、`deploy/` 和 `results/`;新增模型先查 `registry/target_models.yaml` 与 `registry/models.yaml`.
2. 确定任务为编译部署与展示示例、独立正式评价或诊断.报告改动文件、当前证据和未验证项.按仓库规范在修改代码前确认计划,已有明确批准不重复询问.
3. 部署时读取 [deployment.md](references/deployment.md);评价时读取 [evaluation.md](references/evaluation.md);追溯外部设计时读取 [sources.md](references/sources.md).不要默认加载全部参考.

## 执行边界

- `deploy/compile.sh` 完成导出、量化、编译和部署,随后自动调用板端同名入口,用固定少量输入检查通路并生成 README 示例.只打印关键步骤与成功/失败信息,不计算或发布精度、性能、峰值 RSS 报告.不保留旧 run.sh 或快速参数兼容入口.
- `deploy/evaluate.sh` 是独立正式评价入口.容器端仅对已有浮点模型准备同协议参考和输入;板端 evaluate.sh 使用已有 DLA 与程序评价精度、统计性能和进程峰值 RSS.不触发导出、量化、编译、构建或重新部署.
- 每模型的 results/ 只能有 summary.json,所有正式精度、性能、RSS、实际范围、版本与缺失项写入这一个文件.不增加按阶段/运行编号划分的报告、多份 summary 或 Markdown/CSV 指标副本.编译阶段不创建或改写正式报告.
- README 必需的少量输入、结果图及文本放 examples/.预测、日志、缓存和中间指标放日期临时目录,失败现场也不放 results.模型契约、tokenizer 和评价输入属于必要资源,不能为了文件少而删掉运行必需数据.
- 入口直接顺序调用命令,只补充必要阶段提示,工具输出和错误直接显示在终端.不默认增加日志文件、命令包装或额外管理机制;保持整个流程简单干净.
- 主流程独立文件默认无参数运行:主机读取所属模型 config.yaml,板端读取部署包 config.json.入口不重复传配置路径和运行编号.只保留影响行为的参数,不预先补充尚未需要的功能;普通容器步骤使用已配置的 python,板端保留现有 python3.
- Docker 公共环境由创建容器入口和本地配置管理,模型脚本不重复配置 SDK、解释器或 Python 缓存策略.范围和运行编号属于单次执行状态,在入口确定;板端临时目录复用公共工具推导.板端独立运行所需的缓存控制仅放在板端分支,不要误当作 Docker 配置重复设置.
- 临时文件也要精简:只保存运行、评分或排错必需的数据,不默认创建每步状态文件、逐样本日志、多种格式的同义输出或历史副本.逐样本结果优先汇总保存;保留运行器必需的张量、预测和解析日志,新增文件必须有具体用途.
- 两阶段使用相同模型版本、预后处理与 DLA.若契约不能容纳完整评价则停止,不截题、丢样本或在评价中自动重编译.无正式精度协议时报告 partial 和原因,生成样例不得冒充全量精度.
- 未明确要求的测试、耗时任务和大文件下载先确认再执行.修改文件不等于允许执行模型.升级、刷机、重启和停止其他进程按维护授权执行.
- 编译使用 Ubuntu Docker,真实 NPU 验证在目标板执行.先读取本地配置并核对当前环境,不把公开配置模板当作实时设备状态.
- 主机名、地址、凭据和内部绝对路径只保存在本地配置及本地运行记录.正式模型和数据集存 NAS,临时文件使用仓库约定的日期目录.

## 部署与交付

按任务选择必要阶段,复用来源明确且契约匹配的已有产物.

1. **来源与协议:** 核实官方/原作者源码、固定版本、权重 URL 及许可证.Qualcomm 等厂商只能作为 `delivery_reference`.资源记录具体下载 URL、版本、文件名和用途,公开说明不要堆叠哈希.分别核实源码、权重、数据集许可;未核实来源保持 pending.
2. **环境与可行性:** 读取平台配置和本地覆盖,核对 SDK、Converter、NCC、AArch64 编译器、Runtime 及硬件.算子支持文档受权限限制时说明待核实,不编造支持结论.
3. **导出与量化:** 明确形状、dtype、layout、预处理、输出语义及后处理.先确认浮点参考,再校准量化.记录实际 W/A 位宽及原生 I/O,不能混淆 FP16/INT16 或将全部任务统一成 W8A8.
4. **DLA 与板端程序:** G720 普通模型复用 `tools.deploy.compile_dla` 的 MDLA 5.3 配方,特殊链路按模型实现.程序和库须匹配 AArch64 及 Runtime,禁止跨平台覆盖系统库.
5. **编译与示例:** 不运行数据集评分、完整浮点参考或专项测速.直接由少量真实预测生成示例,不要让渲染依赖先完成精度评价.进度信息不冒充质量验收.
6. **独立评价:** 用户要求后才执行.核对完整覆盖和同协议参考,统计纯 NPU/可用端到端性能、LLM Prefill/Decode 和进程 RSS,明确计时及内存范围.缺参考时差值留空,缺实测时写未完成,不补数字.
7. **单报告与文档:** 只由 evaluate.sh 发布 results/summary.json,成功后结果目录不留其他文件,不混用版本或借用旧指标.核实后同步文档和注册状态.新增模型从 templates/model 创建,遵循 tools/check_registry.py 的双入口与单报告约束.

## 模型路由

调用前重新读取当前文件,不要将单模型经验直接外推到新平台.

| 场景 | 仓库参考 | 保留的差异 |
| --- | --- | --- |
| 检测 | `models/perception/object_detection/yolov5s`、`yolov8n` | COCO bbox、letterbox、类别映射、阈值和 NMS |
| 分类 | `models/perception/image_classification/vit_base_patch16_224` | 官方权重版本、归一化、标签顺序和 Top-1/Top-5 |
| 姿态 | `models/interaction/pose_detection/rtmpose_body2d` | WholeBody 133 点、人体框、仿射裁剪、SimCC 和 OKS-NMS |
| 人脸 | `models/interaction/face_recognition/mobilefacenet` | 五点对齐、128 维特征及 LFW 十折阈值协议 |
| 分割与深度 | `models/navigation/segmentation/fastsam`、`models/navigation/single_camera_depth/depth_anything_v2_small` | 类别无关 AR@100 与 DA-2K 点对准确率,不能替换为其他任务指标 |
| 音频 | `models/audio/ambient_sound/yamnet`、`models/audio/stt/whisper_tiny` | 标签投影和评价折;双 DLA、自回归解码与文本规范化 |
| 原生 LLM | `models/gen_ai/llm/minicpm5_2b`、`deepseek_r1_distill_qwen_1_5b` | Prefill/Decode、KV Cache、tokenizer、embedding、上下文;生成 Demo 不等于正式 MMLU/PPL |

G720 结果仅证明对应模型、版本和环境.G5100 无实测时保持未验证,先核实目标平台参数,不能复用 G720 编译目标或指标.

## 收尾

中文说明改了什么、实际执行了哪些检查、未运行哪些任务、兼容性影响及剩余问题.区分文件审阅、静态检查、小批量和全量实测.编译成功或版本查询不等于真实推理.给出中文 Git 提交说明,用户未要求时不自行提交.
