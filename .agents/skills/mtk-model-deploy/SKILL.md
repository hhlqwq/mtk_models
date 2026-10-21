---
name: mtk-model-deploy
description: 在 mtk_models 仓库适配和部署官方开源模型到 MediaTek Genio,处理 NeuroPilot 转换量化、DLA 编译、板端小批量验证及独立全量精度评价.用于新增模型、修复 MTK 部署链路与排查精度差异,不用于其他厂商部署或将历史结果冒充新实测.
---

# MTK 模型部署

交付可复用的模型实现、明确的张量契约、独立板端部署包和对应运行证据.优先复用当前仓库工具,不要为单个模型另造环境管理或流水线框架.

## 开始工作

1. 读取适用的 `AGENTS.md`,检查 `git status --short`,保留用户改动.只读定位模型 README、配置、模型卡、来源、`deploy/` 和 `results/`;新增模型先查 `registry/target_models.yaml` 与 `registry/models.yaml`.
2. 确定任务为部署适配、小批量验证、全量精度评价或诊断.报告改动文件、当前证据和未验证项.按仓库规范在修改代码前确认计划,已有明确批准不重复询问.
3. 部署时读取 [deployment.md](references/deployment.md);评价时读取 [evaluation.md](references/evaluation.md);追溯外部设计时读取 [sources.md](references/sources.md).不要默认加载全部参考.

## 执行边界

- `deploy/run.sh` 为部署入口,板端对应 `bash run.sh`,无参数默认固定小批量验证.不接受旧快速参数,不自动进入全量.
- `deploy/evaluate.sh` 为独立精度入口.容器端仅对已有浮点模型生成全量参考和评价输入;板端对应 `bash evaluate.sh`,仅对已有部署包全量推理、计算精度并保存预测.不调用导出、校准、量化、编译、构建、打包、上传、Demo 或性能专项流程.
- 两种范围使用同一权重、预后处理、量化配方及 DLA,只改变评价样本范围.若现有契约不能容纳全量输入,报告阻塞,不能截题、丢样本或在评价中重新编译.
- 未明确要求的测试、耗时任务和大文件下载先确认再执行.修改文件不等于允许执行模型.升级、刷机、重启和停止其他进程按维护授权执行.
- 编译使用 Ubuntu Docker,真实 NPU 验证在目标板执行.先读取本地配置并核对当前环境,不把公开配置模板当作实时设备状态.
- 主机名、地址、凭据和内部绝对路径只保存在本地配置及本地运行记录.正式模型和数据集存 NAS,临时文件使用仓库约定的日期目录.

## 部署与交付

按任务选择必要阶段,复用来源明确且契约匹配的已有产物.

1. **来源与协议:** 核实官方/原作者源码、固定版本、权重 URL 及许可证.Qualcomm 等厂商只能作为 `delivery_reference`.资源记录具体下载 URL、版本、文件名和用途,公开说明不要堆叠哈希.分别核实源码、权重、数据集许可;未核实来源保持 pending.
2. **环境与可行性:** 读取平台配置和本地覆盖,核对 SDK、Converter、NCC、AArch64 编译器、Runtime 及硬件.算子支持文档受权限限制时说明待核实,不编造支持结论.
3. **导出与量化:** 明确形状、dtype、layout、预处理、输出语义及后处理.先确认浮点参考,再校准量化.记录实际 W/A 位宽及原生 I/O,不能混淆 FP16/INT16 或将全部任务统一成 W8A8.
4. **DLA 与板端程序:** G720 普通模型复用 `tools.deploy.compile_dla` 的 MDLA 5.3 配方,特殊链路按模型实现.程序和库须匹配 AArch64 及 Runtime,禁止跨平台覆盖系统库.
5. **小批量验证:** 两端选同一固定子集,记录标识、范围和实际覆盖.默认部署只做小批量,保留真实预测及少量效果示例.不改变校准配方,不自动跑全量.
6. **独立全量精度:** 用户要求后才执行.检查已有模型及评价资源,计算同协议指标并保存预测.缺少匹配参考可以报告绝对精度,差值留空并注明原因.
7. **文档与状态:** 按实际变化同步模型文档、机器报告、模型总览及注册状态.只归档已核实结果,小批量不得覆盖正式全量指标.

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
