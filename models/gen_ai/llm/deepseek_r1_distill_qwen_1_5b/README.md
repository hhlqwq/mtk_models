# DeepSeek-R1-Distill-Qwen-1.5B

本次与 MiniCPM5 统一的主机质量入口为 `models/gen_ai/llm/evaluate_quality.py`:
同一固定 WikiText2 测试集的前 128 行文本,各自使用官方 tokenizer,
不额外添加特殊 Token,128 Token 分块并重置上下文,每块评分 127 次预测.
末尾不足一块的 Token 数写入报告.比较各模型自己的 FP32 与 W4A16 PPL 变化,
不同词表的绝对 PPL 不直接作为模型排名.原有 26 Token 数值检查与新协议分开,
新的统一量化评价已完成,不沿用旧 FP16 结果代表 W4A16.

统一协议 FP32 参考 PPL 为 **118.70850**,评分 **7620** 个 Token;
主机 W4A16 PPL **193.93135**,相对 FP32 增加 **63.37%**,见 [统一质量报告](results/quality_summary.json).
原生 W4A16 的同协议板端 PPL 使用共享 `evaluate_board_quality.py`,
读取实际 INT16 图契约并通过现有 `neuron_bridge.cpp` 在 MDLA 硬件执行,
不沿用旧 FP16 的 NLL 结果.
同一测试子集实际 MDLA PPL 为 **205.99682**,相对 FP32 增加 **73.53%**.
60 个块的硬件推理合计 **19.94 秒**,评价含 CPU NLL 计算共 **32.94 秒**,
没有 CPU 模型回退.板端、浮点输入 Token 已逐项核对一致;
主机 TFLite 与板端结果均已核对,当前 W4A16 不能因双语 Demo 完成而认定质量合格.

SDK 浮点诊断: 默认注意力 mask=-100 时 PPL 为 124.89242;
仅在诊断进程中改为 -10000 后为 118.71525,与同协议官方 GPU FP32 118.71518 接近.
因此转换差距还包含 mask 的实现兼容问题,不能全部归因为权重量化.
该诊断未修改安装的 SDK、官方权重或已有交付 DLA; 当前 INT16 图不能直接沿用此浮点修改.

原生交付使用两个 DLA: `1024c/prompt.dla` 执行 Prefill,`1024c/decode.dla` 执行 Decode.
两阶段性能单位统一为 token/s. Prefill 按实际输入 Token 数除以 Prompt 耗时计算,保留原始耗时,不作为 TTFT.

中英文文本问答与推理模型.使用 DeepSeek 官方权重,参考地瓜 RDK S 系列的选型与指标展示,不使用厂商预编译模型作为转换输入.

2026-10-10 已完成官方资源校验、静态 ONNX 导出、FP16 编译、Genio 720 双语 Demo 与三后端样例数值检查,状态为 `board_verified`.
原生 W4A16 与 128 Token Prefill 已完成板端双语推理验证; 正式质量与 Genio 5100 尚未验证,不标记为 `complete`.

## 来源与配置

- [官方模型](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B),固定 revision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.
- [地瓜 Model Zoo](https://github.com/D-Robotics/rdk_model_zoo_s) 与 [S100 LLM 工具链](https://developer.d-robotics.cc/rdk_s_doc/Advanced_development/toolchain_development/LLM_Toolchain/s100_LLM_Toolchain) 包含该模型.
- 2026-10-10 核对 [MTK 公开生成式模型列表](https://genio.mediatek.com/doc/iot-aihub/ai_hub/model_zoo/litert_gai/supported_models.html),未列出该模型; 结论仅覆盖公开列表.
- 规格与许可见 [模型卡](model_card.md); 所有官方资源的地址、实际大小与 SHA-256 见 [来源](models/source_url.txt) 和 [资源清单](models/source_manifest.json).

FP16 基线配置为 batch=1、上下文 1024、FP16.28 个 Decoder 分片与 10 个词表投影分片全部通过 Neuron Runtime 硬件接口运行,编译参数为 `--arch=mdla5.3 --suppress-input --suppress-output --disallow-bridge`.
CPU 负责 embedding 查表、RoPE、KV Cache 管理、Greedy 选 Token 和统计; Attention、归一化、MLP 与词表投影在 NPU 执行.不存在模型计算失败后自动回退 CPU 的路径.

Prefill 当前逐 Token 执行,与 Decode 复用同一组分片; 不具备并行 Prefill 优化.注意力在矩阵乘法前缩放 Query,避免 FP16 点积中间值溢出,数学公式不变.
RMSNorm 先按输入幅度归一再计算方差,同时按同一比例调整 epsilon,
避免大幅度隐藏状态在 FP16 平方时溢出,保留官方归一化公式与参数.
本实现通过自有 Runtime 桥接库运行,不调用 `llm_cmdline_tool`,也不声称兼容其 Prompt / Generative 打包格式.

## 编译与部署

在 Ubuntu89 执行,目录为 `/data/users/hailong.he/github/mtk_models`.沿用 `hhl_g720_8011` 容器,不替换容器原有 Python 包.

官方资源目录 `/tmp/hailongcodex/2026-10-10/deepseek_source` 必须同时存在于宿主机和容器,包含资源清单中的六个文件与 `source_manifest.json`.
宿主机目录不在容器挂载范围时,先使用 `docker cp` 复制; 导出前会再次验证每个文件的哈希.

隔离环境 `/tmp/hailongcodex/2026-10-10/deepseek_env` 使用 `--system-site-packages`,仅补充 `transformers==4.44.2`、`safetensors==0.4.5`、`tokenizers==0.19.1`.既有 Torch 为 `2.0.0+cu118`,ONNX 为 `1.13.1`,MTK Converter 为 `8.16.0`.
环境中的绝对路径可通过 `deploy/run.sh` 顶部配置或同名环境变量覆盖,资源地址已归档.

重建隔离环境时执行以下命令.安装只作用于该 venv,沿用容器原有其余依赖:

```bash
docker exec hhl_g720_8011 python3 -m venv --system-site-packages \
  /tmp/hailongcodex/2026-10-10/deepseek_env
docker exec hhl_g720_8011 /tmp/hailongcodex/2026-10-10/deepseek_env/bin/python \
  -m pip install --no-deps transformers==4.44.2 safetensors==0.4.5 tokenizers==0.19.1
```

```bash
cd /data/users/hailong.he/github/mtk_models
bash models/gen_ai/llm/deepseek_r1_distill_qwen_1_5b/deploy/run.sh
```

脚本依次校验资源、导出、生成 PyTorch/ONNX 参考、编译 MDLA 5.3 DLA、交叉编译桥接库,并上传到 `/root/hailong.he/open_models/deepseek_r1_distill_qwen_1_5b`.
运行时间较长,逐分片显示进度,每阶段保留日志; 任一步骤失败即停止.权重及生成产物不提交普通 Git.

当前临时实验的编译产物在 Docker `/tmp/hailongcodex/2026-10-10/deepseek_models`,源代码试验副本在宿主机与 Docker `/tmp/hailongcodex/2026-10-10/deepseek_code`.正式脚本默认输出为模型目录下 `models/generated`.

## 原生 GAI W4A16 交付流程

2026-10-10 已完成原生 GAI W4A16 转换、MDLA5.3 编译及 `llm_cmdline_tool` 板端验证.
原生结果与 FP16 基线分别记录,正式量化质量尚未验收.

工具来源为 MTK 官方 `GAI-Deployment-Toolkit-v2.0.8_qwen2.5-0.5b-1.5b-7b-v0.1.tar.gz`,
内含 `mtk_llm_sdk==2.7.5`.用户下载的原始归档位于
`/data/users/hailong.he/data/MTKG720/GAI_Toolkit/`,大小 `571891487` 字节,
SHA256 为 `da10e770e2950542ab17c63182b0b932348ab05ffdceb18cbb097555ca6127f5`.
下载地址见 [来源记录](models/source_url.txt).厂商工具及其中的第三方源码不纳入本仓库.

工具解压至宿主机 `/tmp/hailongcodex/2026-10-10/gai_toolkit/`,
容器副本为 `/tmp/hailongcodex/2026-10-10/gai_toolkit_qwen25/`.
独立环境 `/tmp/hailongcodex/2026-10-10/deepseek_gai_env` 继承容器系统包,
原生实验目录为 `/tmp/hailongcodex/2026-10-10/deepseek_native`.

在既有容器中创建独立环境,安装本地工具包和兼容依赖:

```bash
docker exec hhl_g720_8011 python3 -m venv --system-site-packages \
  /tmp/hailongcodex/2026-10-10/deepseek_gai_env
docker exec hhl_g720_8011 /tmp/hailongcodex/2026-10-10/deepseek_gai_env/bin/python \
  -m pip install --no-index --no-deps \
  /tmp/hailongcodex/2026-10-10/gai_toolkit_qwen25/mtk_llm_sdk_v2.7.5/mtk_llm_sdk-2.7.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
docker exec hhl_g720_8011 /tmp/hailongcodex/2026-10-10/deepseek_gai_env/bin/python \
  -m pip install transformers==4.44.2 safetensors==0.4.5 tokenizers==0.19.1 \
  sentencepiece==0.2.0 datasets==2.21.0 evaluate==0.4.3 \
  nvidia-ml-py3==7.352.0 pyarrow==17.0.0
```

PyArrow 固定为 17.0.0,与当前 NumPy1.26.4 兼容,不升级容器全局 NumPy.

`deploy/host/prepare_native.py` 重新校验官方资源,生成独立配置,使用官方
Chat Template 编码双语校准输入,不修改源权重.转换配置采用官方 tokenizer
的 BOS `151646`、EOS `151643`,保留 RoPE `10000`.
上下文先限制为 1024,Prefill 为 128,Decode 为 1; 配置与静态图必须同步生成.
16 条自编双语校准输入仅用于部署流程检查,不代表正式精度评测.

在 Ubuntu89 执行,脚本及资源路径必须在容器中可见:

```bash
cd /data/users/hailong.he/github/mtk_models
bash models/gen_ai/llm/deepseek_r1_distill_qwen_1_5b/deploy/run_native.sh
```

可单独运行 `prepare`、`calibrate`、`quantize`、`shape`、`compile` 阶段,
默认通过 Converter 后端执行 `asym4W_sym16A` 和 Hessian 权重优化.
编译调整为 G720 的 MDLA5.3、L1 256KB、单核,保留原生脚本优化选项并禁止桥接.
已验证 tokenizer、INT16 embedding、板端 Prompt/Decode 接口和双语输出.
两个固定样例的原生文本输入与官方 Token 输入输出一致.

编译通过后,在容器内执行 `deploy/host/package_native.py --work <原生实验目录>
--output <新的交付目录>`,生成两个 DLA、INT16 embedding、BPE tokenizer、
Yocto 配置、双语 Prompt 与运行脚本.打包前读取静态 TFLite I/O 验证 INT16 契约,
交付清单记录文件哈希,初始状态为 `compiled_pending_board`.

## 板端测试与汇总

在 192.168.0.92 / Genio 720 执行:

```bash
EVAL_RUN_ID=20261010_deepseek_fp16_v3 bash /root/hailong.he/open_models/deepseek_r1_distill_qwen_1_5b/run.sh
```

默认测试包含中英文各一个对话 Demo 和一个自编短文本质量样例.保留官方 Tokenizer、Chat Template 和 EOS,固定 Greedy 解码,最多生成 512 Token.
通过教师强制的逐 Token NLL 对比官方 PyTorch FP32、ONNX FP32 与 NPU FP16; 自编短文本 PPL 仅用于数值检查,不能当作正式基准精度.
编译前可通过 `EVAL_CORPUS` 指定 UTF-8 JSONL,每行含唯一 `id` 与 `text`,语料来源和评测协议需另外归档.

结果存于板端 `results/<运行编号>`; 先回收至编译主机,再复制到 Docker 中并运行:

```bash
/tmp/hailongcodex/2026-10-10/deepseek_env/bin/python \
  /data/users/hailong.he/github/mtk_models/models/gen_ai/llm/deepseek_r1_distill_qwen_1_5b/deploy/host/summarize.py \
  --source /tmp/hailongcodex/2026-10-10/deepseek_source \
  --models /tmp/hailongcodex/2026-10-10/deepseek_models \
  --results /tmp/hailongcodex/2026-10-10/deepseek_results/20261010_deepseek_fp16_v3
```

汇总生成 `summary.json` 与 `examples/output/*.txt`,记录原始 Token、实际文本、截断、EOS、TTFT、Prefill / Decode 速度、纯 NPU 调用耗时与峰值 RSS.
模型加载、产物校验单独计时,TTFT 从处理 Prompt 开始; Decode 速度分子为实际新增的模型调用次数,不把首 Token 重复计入.
峰值 RSS 只覆盖进程地址空间中的驻留内存,不能代表 NPU 驱动的全部内存;
同时记录加载前、加载后和推理时最低系统可用内存,其差值仅为本次系统观测.
发生失败时生成 `failure.json`,不能以 NCC 编译成功代替板端验证.

## 实测记录

本次运行为 `20261010_deepseek_fp16_v3`,平台 MT8189 / Genio 720,Neuron Runtime 8.2.16,
系统 Linux 6.6.137,Python 3.12.13,NumPy 1.26.4.实际记录见 [summary.json](results/summary.json),
38 个 DLA 与 embedding 的大小和 SHA-256 见 [产物清单](results/artifact_manifest.json).

| Demo | 生成 Token | TTFT (秒) | Decode (Token/s) | 总耗时 (秒) | EOS / 最终回答 |
| --- | ---: | ---: | ---: | ---: | --- |
| [中文实际输出](examples/output/zh_demo.txt) | 73 | 3.245 | 3.78 | 22.292 | 均完成 |
| [英文实际输出](examples/output/en_demo.txt) | 412 | 3.112 | 3.81 | 111.115 | 均完成 |

原始框架文本见 [中文 PyTorch](examples/output/zh_demo_pytorch.txt) 与 [英文 PyTorch](examples/output/en_demo_pytorch.txt).
生成文本保留真实思考过程和特殊 Token,不人工修饰重复语句.本次两个 Demo 均未截断.

26 个自编双语质量 Token 的 PPL: PyTorch **144.5709**,ONNX **144.5712**,NPU **145.0466**.
ONNX / NPU Argmax 一致率为 **96.15%**,最大逐 Token NLL 差异为 **0.03532**;
ONNX / PyTorch 最大逐 Token NLL 差异约 **0.000014**.这些值只用于样例数值检查,不是正式精度结论.

纯 NPU 调用平均耗时为 **220.11 ms/Token**.进程峰值 RSS 为 **148.10 MiB**,
加载前系统可用内存 **6716.26 MiB**,推理时最低 **3204.17 MiB**,下降 **3512.09 MiB**.
DLA 加 embedding 的部署资源合计约 **3.56 GB**,因此 RSS 不能作为模型总内存.

适配时修复了两处 FP16 问题: 注意力点积中间值 82550 超过 FP16 上限;
隐藏状态幅度约 759 时 RMSNorm 平方溢出,导致输出层全零.
第一轮完整运行未通过数值检查; 在前置注意力缩放和等价 RMSNorm 缩放后,
第二轮数值检查通过,第三轮使用 512 Token 上限完成英文最终回答.

地瓜参考数据为 S100P、q4、输入长度 256、上下文 1024 时 TTFT 108 ms、39.49 Token/s、内存 1.1 GB,来源为上文 S100 工具链性能表.这些数值不能作为 MTK 实测结果.

## 原生 W4A16 实测与 NAS 交付

运行编号 `20261010_deepseek_w4a16_v2`,使用同一官方权重与 Chat Template.
中文结果沿用已成功的 v1,英文将生成上限调整到 768 后完成; 原始运行编号保留在报告中.
两路 TFLite 的 59 个输入与 57 个输出均为 INT16,量化 Scale 全部为有限正数.

| Demo | 生成 Token | Prefill (秒) | Decode (Token/s) | 总耗时 (秒) | EOS / 最终回答 |
| --- | ---: | ---: | ---: | ---: | --- |
| [中文实际输出](examples/output/zh_demo_native.txt) | 273 | 0.325992 | 16.75 | 18.773 | 均完成 |
| [英文实际输出](examples/output/en_demo_native.txt) | 636 | 0.325769 | 16.75 | 39.827 | 均完成 |

Prefill 时间来自原生 CLI,不包含模型加载或切换,不能当作 TTFT.
Decode 速度按原生 CLI 的模型调用计数,排除首 Token.
进程峰值 RSS 约 813.57 / 816.73 MiB,系统可用内存分别下降 938.87 / 909.03 MiB.
后者仅为本次系统观测,不是驱动内存的独立测量.
两个 DLA 与 embedding 合计 2036961220 字节,完整交付目录约 2.04 GB.

英文样例出现不自然的语法,保留实际文本和原始 Token; 16 条自编校准输入及双语 Demo
不足以证明正式量化质量达标.未对 W4A16 执行正式数据集精度评测,
也不能沿用 FP16 的 PPL 与一致率.原生证据见 [报告](results/native_summary.json)
与 [文件哈希及接口清单](results/native_artifact_manifest.json).

NAS 交付目录:

```text
/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/模型部署/MTK/G720/TFLite/GenerativeAI/LLM/DeepSeek-R1-Distill-Qwen-1.5B/deepseek-r1-distill-qwen-1.5b
```

目录组织与既有 LLM 一致,包含 `1024c/{prompt.dla,decode.dla}`、
`tokenizer/{embedding_int16.bin,vocab.txt,merges.txt,added_tokens.yaml}`、
`scripts/{config-yocto.yaml,run.sh,prompts/}`、`results/`、许可证和文件清单.

板端已部署到 `/root/hailong.he/open_models/deepseek_r1_distill_qwen_1_5b_native`:

```bash
cd /root/hailong.he/open_models/deepseek_r1_distill_qwen_1_5b_native
bash scripts/run.sh
# 使用原生文本分词,输入仍包含官方 Chat Template.
INPUT_MODE=text bash scripts/run.sh
```

默认总生成上限为 768,可通过 `MAX_NEW_TOKENS` 调整; 原生 CLI 的 `-m` 计数
不包含首 Token,脚本自动减 1.上下文 1024、Prefill 128 涉及静态图,
修改时必须重新转换和编译,不能只改配置.脚本保存日志,指标归档另行使用
`package_native.py --board-results <原始板端证据目录>` 校验日志哈希并更新清单.

## 使用限制

小型推理模型可能生成较长思考过程、重复内容或中英文混用.生成上限仍可能截断思考内容,必须查看实际文本与截断标记,不能视为完整回答.
FP16 基线模型体积和内存明显大于 q4 模型,当前不声称达到地瓜的内存或速度.原生 W4A16 已验证推理与 128 Token Prefill,正式质量评测和 Genio 5100 验证尚未完成.
DeepSeek 模型许可为 MIT,Qwen 基础模型采用 Apache-2.0; 分发权重或转换产物时保留许可与归属信息.

2026-10-12 继续精度定位: 使用共享 compare_logits.py 对统一文本前 4 个块的全部 127 个预测位置做三后端 logits 对比.
记录余弦、去均值余弦、RMSE、MAE、Top1 一致率与 KL.原始 INT16 输入与实际输出哈希用于核对.
该诊断仅覆盖 Prefill 图,不能宣称 Decode 或完整精度验收通过.初版实际结果见 [logits 报告](results/logits_summary.json).

修复候选: mask_value=-10000; 当前不启用权重旋转.仅修改独立转换配置,官方权重保持原样.
浮点诊断已核对与官方参考的平均余弦接近 1、Top1 100%; 量化后的效果仍待实际验证.
编译改用 NCC --opt-accuracy,移除会牺牲精度的 --opt-aggressive; Prefill/Decode 仍各一个 DLA.
环境变量 MASK_VALUE、ROTATE_WEIGHTS 控制可追溯的 SDK 设置,可选 WEIGHT_OPT_CONFIG 复用官方累计误差优化.
新候选必须重新完成校准、量化、主机和板端验证,不得覆盖当前交付后直接标为通过.

当前 W4A16 前 4 块、508 个预测位置: FP32→TFLite 平均余弦 0.813886,FP32→MDLA 0.809418,TFLite→MDLA 0.962701.
FP32→MDLA Top1 一致率 43.70%,存在明显差异,当前输出精度未合格.

Decode 诊断新增 --decode-steps: 128 Token Prefill 后继续输入真实后续 Token,各后端沿用自己的 KV.
缓存布局按 SDK 掩码的有效后缀核对,缓存转移按实际 I/O 量化参数重编码.
共享工具分别记录 Prefill 与 Decode 的输出比较,不按语言拆分,诊断通过也不代表完整验收.

板端 Decode 实测 mask 逻辑长度 2050 字节,Runtime 要求 2064 字节.
仅对已确认的单行 mask 补齐 16 字节尾部对齐,填充区保持屏蔽;其他输入输出大小仍严格匹配.

原版本 Decode 实测: 4 段固定上下文,每段继续 16 个真实 Token,共 64 个输出位置.
FP32→MDLA 平均余弦 0.823447,Top1 一致率 56.25%; TFLite→MDLA 平均余弦 0.968509.
测试已执行使用 evaluated 标记,精度未验收使用 verified=false、quality_qualified=false,避免混淆.

打包支持 --quantized-prefix 指定本次任务中带官方优化配置后缀的量化目录,仍要求独立空目录和两个 DLA.

修复版静态 Prefill 图已确认 mask scale=0.30517578125,可表达 -10000; 输入准备会拒绝无法表达配置掩码的图,避免静默裁剪.

原生固定样例使用共享 evaluate_native.py,从各自交付清单读取停止 Token,避免两款模型 EOS 混用.
新包包含独立 scripts/evaluate_native.py,运行日志与汇总性能保留在包内 results.

新候选的统一性能可用 --performance-input 指定固定 128 个真实 Token,最多生成 128 Token.
性能独立于双语功能样例,统一为一行 Prefill/Decode token/s,不使用填充位置冒充有效输入.

掩码修复 W4A16 第二版已完成两个 DLA 的三后端诊断: Prefill 平均余弦 FP32→MDLA 0.894111,Decode 0.899299.
4 块诊断的 FP32 PPL 93.38109、MDLA PPL 132.16824,相对增加 41.54%,仍不合格.
该诊断前缀不能与上文 60 块完整统一子集的 PPL 混用.原版本和修复候选的原始报告均保留.

SDK 旋转输出目录带 _rotate_hadamard_0 后缀,Shape 与编译入口按相同配置定位该目录.
本轮 DeepSeek 未启用旋转; 此处仅补齐可选旋转目录的定位规则.

下一轮使用全 W8A16: 设置 ATTENTION_PRECISION=sym8W_sym16A,MLP_PRECISION 与 TAIL_PRECISION 默认跟随,保持 MASK_VALUE=-10000.
复用相同配置下已完成的 32 批校准数据,新建独立工作目录.量化成功后生成 quantization_settings.json,打包清单记录 Attention/MLP/LM Head 的真实精度.
仍只编译 Prefill 128 Token 和 Decode 1 Token 两个 DLA,上下文 1024.全 W8 编译不再使用专属 W4 Decode 拆分选项.

量化统一通过共享 quantize_native.py 传播 SDK 转换异常,并在成功标记前核对 29 个非空分片及逐片 SDK 元数据.兼容逻辑仅用于本次进程,保留已完成的全 W8 候选.

全 W8 第一版的 Prefill/Decode FP32→MDLA 平均余弦为 0.987803/0.991775,Prefill 仍有额外误差.温和编译未改善该误差,候选证据已归档.
后续 -4096 掩码在完整 60 块统一子集的 SDK FP32 PPL 为 118.70850,与官方 FP32 相同;正在重新生成校准数据并保留全 W8 方案.该浮点检查尚不代表新量化候选达标.

新诊断补齐决定首个生成 Token 的 Prefill 最后一行,四块共核对 512 个输出位置,另报告最后一行余弦;PPL 仍只评分每块 127 次下一 Token,Decode 保留 64 个真实后续位置.

## 全 W4A16 优化方向 (2026-10-12)

根据最新确认,最终候选的 Attention、MLP、LM Head 均采用 `asym4W_sym16A`,不采用 W8 或 W8/W4 混合精度交付.此前 W8 产物仅保留为诊断参考.

新候选使用独立目录,旋转为 Hadamard seed=0.共享 `../prepare_calibration.py` 从已校验的 WikiText2 train 中按 seed=17 采样 64 段各 512 Token,保留原短对话,重新生成 128 批校准;测试集不参与校准.DeepSeek 使用已核实的 mask=-4096,MiniCPM5 保持 mask=-100.继续使用 Hessian 和累计层误差参数优化.

保持 Prefill=128、上下文=1024、Decode=1,每款仅两个 DLA.新候选仍须三后端输出、统一 WikiText2 PPL 和板端性能验证,当前不代表精度合格.

统一精度也逐层核对 SDK 实际权重分组,必须确认 Attention、MLP、LM Head 全部符合请求的 W4A16,成功后保存 `weight_precision_audit.json`;仅凭文件名不认定实际精度.

交付清单同步归档统一精度逐层核对结果,打包核对 Attention、MLP、LM Head 配置与分片数量,保留历史包兼容性.
