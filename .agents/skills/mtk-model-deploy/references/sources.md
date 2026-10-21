# 参考来源

下列开源 skill 于 2026-10-21 查阅.本技能结合本仓库独立编写,不复制厂商实现或将厂商参数改名后当作 MTK 命令.后续使用时重新核对来源版本.

| 来源 | 采用的思路 |
| --- | --- |
| [NVIDIA DeepStream import skill](https://github.com/NVIDIA/DeepStream/blob/main/skills/deepstream-import-vision-model/SKILL.md) | 任务分流、张量和预后处理契约、以真实运行及语义证据判定就绪 |
| [NVIDIA TAO RT-DETR skill](https://github.com/NVIDIA/skills/blob/main/skills/tao-train-rtdetr/SKILL.md) | 通用入口与特定部署说明分开,参考按阶段读取 |
| [Hugging Face evaluation skill](https://github.com/huggingface/skills/blob/main/skills/huggingface-community-evals/SKILL.md) | 小样本先行、明确后端及评价协议;本项目另设独立全量入口 |
| [Hugging Face skills](https://github.com/huggingface/skills) | `SKILL.md` 发现入口与按需参考 |
| [MediaTek EDPA/EDMA 说明](https://genio-community.mediatek.com/t/edpa-in-genio520-720/1902) | Genio 520/720 目标差异,结合本项目编译与 Runtime 诊断 |

MediaTek 工具、系统发布和算子支持资料优先沿仓库 `docs/environment.md`、`docs/genio_720.md` 与 `docs_study/` 中官方入口查询.页面要求 MOL/NDA 登录时标为需授权,不把搜索摘要当作完整支持列表.

项目事实以当前 AGENTS.md、配置、模型总览、注册表及模型源码/来源/结果为准.公共复用点为 tools/deploy.py、tools/board.py、tools/evaluation.py、tools/summary.py、tools/render_examples.py 与 common/cpp/evaluation.h.采用外部 skill 的分阶段思路时必须遵守本项目 compile.sh/evaluate.sh 双入口和单一 summary.json 约定.

README 数字和注册状态仅用于定位已有证据,不能证明当前设备状态或新运行成功.正式移植来源不明确时保持 pending,Qualcomm 等交付参考不能替代官方模型源.
