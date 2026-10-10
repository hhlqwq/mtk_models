# DeepSeek-R1-Distill-Qwen-1.5B

中英文文本问答与推理模型.使用 DeepSeek 官方权重,参考地瓜 RDK S 系列的选型与指标展示,不使用厂商预编译模型作为转换输入.

2026-10-10 已完成官方资源校验、静态 ONNX 导出、FP16 编译、Genio 720 双语 Demo 与三后端样例数值检查,状态为 `board_verified`.
正式基准精度、W4A16、并行 Prefill 和 Genio 5100 尚未验证,不标记为 `complete`.

## 来源与配置

- [官方模型](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B),固定 revision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.
- [地瓜 Model Zoo](https://github.com/D-Robotics/rdk_model_zoo_s) 与 [S100 LLM 工具链](https://developer.d-robotics.cc/rdk_s_doc/Advanced_development/toolchain_development/LLM_Toolchain/s100_LLM_Toolchain) 包含该模型.
- 2026-10-10 核对 [MTK 公开生成式模型列表](https://genio.mediatek.com/doc/iot-aihub/ai_hub/model_zoo/litert_gai/supported_models.html),未列出该模型; 结论仅覆盖公开列表.
- 规格与许可见 [模型卡](model_card.md); 所有官方资源的地址、实际大小与 SHA-256 见 [来源](models/source_url.txt) 和 [资源清单](models/source_manifest.json).

当前配置为 batch=1、上下文 1024、FP16.28 个 Decoder 分片与 10 个词表投影分片全部通过 Neuron Runtime 硬件接口运行,编译参数为 `--arch=mdla5.3 --suppress-input --suppress-output --disallow-bridge`.
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

## 使用限制

小型推理模型可能生成较长思考过程、重复内容或中英文混用.生成上限仍可能截断思考内容,必须查看实际文本与截断标记,不能视为完整回答.
FP16 基线模型体积和内存明显大于 q4 模型,当前不声称达到地瓜的内存或速度.W4A16、并行 Prefill、正式质量评测和 Genio 5100 验证尚未完成.
DeepSeek 模型许可为 MIT,Qwen 基础模型采用 Apache-2.0; 分发权重或转换产物时保留许可与归属信息.
