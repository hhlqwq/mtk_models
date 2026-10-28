# DeepSeek-R1-Distill-Qwen-1.5B

> 既有 Python 实现已完成板端生成检查和固定小样本回归.当前 16 段训练校准候选已完成软件小样本对照和双 DLA 编译,尚未上传或进行新候选板端验证;全量精度、标准性能及 RSS 尚未评价.

使用 [DeepSeek 官方权重](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B),面向 Genio 720 部署全 W4A16 模型.来源与许可见[模型卡](model_card.md)和[资源记录](models/source_url.txt).

适配约定、精度问题及修复依据见 [ADAPTATION.md](ADAPTATION.md).评价范围与指标见[唯一报告](results/summary.json).

[中文生成片段](examples/output/zh_demo.txt)和[英文生成片段](examples/output/en_demo.txt)均来自当前板端模型,每条最多 32 Token,不代表完整回答或正式精度.

## 编译与部署

模型与数据提前放在 NAS,连接信息和工具链路径由本地配置提供.板端生成与评价使用 Python3、NumPy、tokenizers 和匹配的 Neuron Runtime,不再使用原生 CLI 或自编桥接库.参数统一在 [config.yaml](config.yaml): Prefill=128、Decode=1、上下文默认 1024、校准批次 48.

权重优化使用 Hessian,稳定性系数见 [hessian_stable.json](deploy/host/hessian_stable.json).校准默认读取 NAS 中的 WikiText2 raw train,训练文本路径相对于数据集根目录解析,参数文件路径相对于本模型目录解析.

训练文本段数由 `native.calibration_train_samples` 控制,当前为 16 段,每段固定 256 Token,另保留 8 条双语输入.历史 8 段配方保留原来的均匀取样;新增覆盖从至少 32 个均匀候选位置选取,使 16/32 段对照可复用同一校准池.启用训练文本时,`calibration_batches` 必须为 `2 * (calibration_train_samples + 8)`,例如 16 段对应 48 批次,32 段对应 80 批次.未启用训练文本时使用原有 16 条双语输入和 32 批次.不匹配的配置会在替换转换目录前报错,避免实际只覆盖部分输入.

当前编译配置选择不旋转、16 段混合训练文本校准、Hessian 稳定性 0.1 的待板端复核候选,尚未确认全量质量.[唯一报告](results/summary.json)保留旧无优化基线的小样本实测,不代表当前候选已完成板端评价.候选对照、独立 64 题复核和逐层局部误差见 [适配说明](ADAPTATION.md).

进入已有编译容器,在本模型目录执行:

```bash
bash deploy/compile.sh
```

依次完成模型适配、校准、W4A16 量化、双静态图导出、双 DLA 编译、打包和上传后结束,提示板端部署目录及命令.用户登录开发板后,在部署目录执行 `bash compile.sh`,实时显示双语输入和生成文本.每条最多生成 32 Token 的片段,保存板端 `examples/output/zh_demo.txt` 和 `en_demo.txt`,并对固定子集执行任务精度及数值误差检查,发布 `results/summary_test.json`.总入口完成后自动回传报告和示例,不执行专项性能或 RSS 测量.

## 独立评价

编译后在同一模型目录执行:

```bash
bash deploy/evaluate.sh
```

此步骤只用已有浮点工作副本和静态图准备完整参考,不会重新量化、编译或部署.默认使用 CUDA FP32;需要 CPU 时修改 `evaluation.reference_device`.

- MMLU: test 全部 14042 题、57 学科、0-shot.两端共用题目 Token,按空格加 A/B/C/D 的单 Token logits 评分,不套聊天模板.
- WikiText2: raw test 全部文本,按 128 Token 分块并重置每块上下文,每块评分 127 次下一 Token,记录不足一块的尾部数量.
- 当前固定 MMLU 题集按此格式编码最长 1010 Token,可容纳于默认 1024 上下文.更换输入协议后仍检查完整题目长度,超过已有上下文时在浮点评分前停止并输出 `mmlu/length_report.json`,不截题、不丢题、不自动重编译.

主机 evaluate 入口自动上传全量题目、Token、质量评价资源与浮点参考至板端 `build/reference/full/` 和 `build/reference/full.json`.然后在板端部署目录运行:

```bash
bash evaluate.sh
```

板端核对部署模型与参考输入后,采集生成样例、128 Token 输入/最多 128 Token 输出的Python Prefill/Decode 阶段速度及进程 RSS,再执行完整 MMLU/WikiText2.最终只发布 `results/summary.json`.速度按实际生成数量记录,Python 阶段速度与纯 NPU 调用耗时分开标注,未采集分项明确列出.板端预测与内部指标统一保存在本模型部署目录的 build/work/,每次开始推理时覆盖.

## 模型约定

使用官方 tokenizer 的 Token 输入,保留聊天模板、BOS/EOS 和停止列表.GQA 为 Q/KV Head=12/2,RoPE=10000;输入、KV Cache、mask、RoPE 与输出使用图中实际 INT16 接口.外置 embedding、tokenizer、Prefill/Decode DLA 和配置必须成套使用.Python 使用随包的官方 tokenizer.json 解码,生成输入继续使用已准备的官方 Token.

校准从全文的 32 个均匀候选位置选取 16 段 256 Token 训练文本,并保留 8 条双语输入,测试集不参与校准.每条保存 Prefill 和单 Token Decode 数据,共 48 个批次.固定小样本比较见 [ADAPTATION.md](ADAPTATION.md),完整质量由独立评价确认.

### 输出误差统计

compile.sh 独立对固定少量输入比较浮点与 NPU 输出,在终端显示 CosSim、RMSE、NRMSE 和实际覆盖范围.普通模型最多选择 5 次输入,不减少原有 compile 通路检查样本数;LLM 比较示例首个 Prefill 块的有效位置,BEVFusion 比较单帧的 NPU 子图.这些结果用于数值检查,不能代替任务精度.全量 evaluate.sh 只计算任务精度、性能和 RSS,不保存或比较全量浮点张量.

参考为普通 NPZ 数组文件,保存在编译机器本地日期临时目录并上传板端 build/reference/output_reference.npz,由下次 compile 覆盖,不使用数据库,不写入 NAS.板端中间结果仍使用本模型 build/work/.NRMSE 定义为 100 × RMSE / RMS(浮点参考),多输出分别统计.详见[输出误差统计](../../../../docs/output_metrics.md).

## DLA 文件命名

编译和部署包统一使用模型名、精度、层数、阶段、输入 Token 数与上下文长度命名.例如当前 1024 上下文:

- `DeepSeek-R1-Distill-Qwen-1.5B_asym4W_sym16A_28layer_prefill_128t_1024c.dla`
- `DeepSeek-R1-Distill-Qwen-1.5B_asym4W_sym16A_28layer_decode_1t_1024c.dla`

文件保存在对应的 `1024c/` 目录,加载路径由 `manifest.json` 的 `model_files` 指定.更换上下文或 Prefill 长度时按实际配置命名;历史结果中的旧文件名仅表示当时的产物.

### compile 子集精度检查

compile.sh 使用 100 道覆盖 57 学科的 MMLU 题和 16 个 WikiText2 块,复用正式评价协议.主机执行子集浮点推理,板端执行同一子集 NPU 推理,在终端打印精度和损失,并保存 results/summary_test.json.正的损失表示 NPU 变差,负值表示改善;Top-1、AP、AR 等使用浮点减 NPU 的百分点,WER 使用 NPU 减浮点的百分点,PPL 使用增量及相对增幅.

普通样本跨完整划分按固定间隔抽取,保留展示及数值检查的必要输入;LFW、ESC-50 和 MMLU 保持折、类别和学科覆盖.同一子集两端通过样本指纹核对;MMLU 固定 seed=17 分层抽样.该报告不代表全量成绩,不改写 results/summary.json.中间数据覆盖保存在本模型 build/work/,CosSim、RMSE、NRMSE 仍只比较固定少量输入.详见[输出误差统计](../../../../docs/output_metrics.md).
