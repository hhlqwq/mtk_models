# DeepSeek-R1-Distill-Qwen-1.5B

> 当前 Python 实现已完成板端生成检查和固定小样本回归,全量精度、标准性能及 RSS 尚未评价.

使用 [DeepSeek 官方权重](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B),面向 Genio 720 部署全 W4A16 模型.来源与许可见[模型卡](model_card.md)和[资源记录](models/source_url.txt).

适配约定、精度问题及修复依据见 [ADAPTATION.md](ADAPTATION.md).评价范围与指标见[唯一报告](results/summary.json).

[中文生成片段](examples/output/zh_demo.txt)和[英文生成片段](examples/output/en_demo.txt)均来自当前板端模型,每条最多 32 Token,不代表完整回答或正式精度.

## 编译与部署

模型与数据提前放在 NAS,连接信息和工具链路径由本地配置提供.板端生成与评价使用 Python3、NumPy、tokenizers 和匹配的 Neuron Runtime,不再使用原生 CLI 或自编桥接库.参数统一在 [config.yaml](config.yaml): Prefill=128、Decode=1、上下文默认 1024、校准批次 32.

权重优化使用 Hessian,稳定性系数见 [hessian_stable.json](deploy/host/hessian_stable.json).校准默认读取 NAS 中的 WikiText2 raw train,训练文本路径相对于数据集根目录解析,参数文件路径相对于本模型目录解析.

当前配方是不旋转、混合训练文本校准、Hessian 稳定性 0.1,仍在精度优化中.[唯一报告](results/summary.json)保留旧无优化基线的小样本实测,不代表当前配方已完成评价.候选对照、独立 64 题复核和逐层局部误差见 [适配说明](ADAPTATION.md).

进入已有编译容器,在本模型目录执行:

```bash
bash deploy/compile.sh
```

依次完成模型适配、校准、W4A16 量化、双静态图导出、双 DLA 编译、打包和上传后结束,提示板端部署目录及命令.用户登录开发板后,在部署目录执行 `bash compile.sh`,实时显示双语输入和生成文本.每条最多生成 32 Token 的片段,只检查执行通路并保存板端 `examples/output/zh_demo.txt` 和 `en_demo.txt`,不评分、不测速、不采集 RSS.示例不再由编译服务器自动取回.

## 独立评价

编译后在同一模型目录执行:

```bash
bash deploy/evaluate.sh
```

此步骤只用已有浮点工作副本和静态图准备完整参考,不会重新量化、编译或部署.默认使用 CUDA FP32;需要 CPU 时修改 `evaluation.reference_device`.

- MMLU: test 全部 14042 题、57 学科、0-shot.两端共用题目 Token,按空格加 A/B/C/D 的单 Token logits 评分,不套聊天模板.
- WikiText2: raw test 全部文本,按 128 Token 分块并重置每块上下文,每块评分 127 次下一 Token,记录不足一块的尾部数量.
- 当前固定 MMLU 题集按此格式编码最长 1010 Token,可容纳于默认 1024 上下文.更换输入协议后仍检查完整题目长度,超过已有上下文时在浮点评分前停止并输出 `mmlu/length_report.json`,不截题、不丢题、不自动重编译.

将模型资源目录 `build/native/evaluation/full/` 的内容单独放入已有板端部署包 `evaluation/full/`,然后在板端部署目录运行:

```bash
bash evaluate.sh
```

板端核对部署模型与参考输入后,采集生成样例、128 Token 输入/最多 128 Token 输出的Python Prefill/Decode 阶段速度及进程 RSS,再执行完整 MMLU/WikiText2.最终只发布 `results/summary.json`.速度按实际生成数量记录,Python 阶段速度与纯 NPU 调用耗时分开标注,未采集分项明确列出.预测与内部指标保存在日期临时目录.

## 模型约定

使用官方 tokenizer 的 Token 输入,保留聊天模板、BOS/EOS 和停止列表.GQA 为 Q/KV Head=12/2,RoPE=10000;输入、KV Cache、mask、RoPE 与输出使用图中实际 INT16 接口.外置 embedding、tokenizer、Prefill/Decode DLA 和配置必须成套使用.Python 使用随包的官方 tokenizer.json 解码,生成输入继续使用已准备的官方 Token.

校准使用全文均匀取出的 8 段 256 Token 训练文本和 8 条双语输入,测试集不参与校准.每条保存 Prefill 和单 Token Decode 数据,共 32 个批次.固定小样本比较见 [ADAPTATION.md](ADAPTATION.md),完整质量由独立评价确认.
