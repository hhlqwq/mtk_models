# DeepSeek-R1-Distill-Qwen-1.5B 模型卡

## 开源上游

- 作者与官方权重: [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).
- 固定权重仓库 revision: `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.
- 官方项目: [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1).
- 基础模型: `Qwen/Qwen2.5-Math-1.5B`.
- 本模型代码与权重许可: MIT; Qwen 基础模型许可: Apache-2.0.
- 权重格式: BF16 Safetensors.已下载固定版本,实际大小为 3554214621 bytes.
  每个资源的 SHA-256 见 [资源清单](models/source_manifest.json).
- 原始实现: Hugging Face Transformers 的 `Qwen2ForCausalLM`.
  本次隔离环境使用 Transformers `4.44.2`,官方配置中的版本字段仅为保存元数据.

## 模型规格

根据官方 `config.json`:

| 属性 | 值 |
| --- | --- |
| 规模标识 / 实际参数量 | 名称为 1.5B; 官方 Safetensors Tensor 形状合计 1,777,088,000 |
| 类型 | 纯文本 Causal LM,`model_type=qwen2` |
| 层数 / Hidden Size | 28 / 1536 |
| Attention Heads / KV Heads | 12 / 2,GQA |
| Intermediate Size | 8960 |
| Vocab Size | 151936 |
| RoPE Theta | 10000 |
| Max Position Embeddings | 131072,不代表板端上下文容量 |
| Embedding / LM Head | `tie_word_embeddings=false` |
| 本次实际配置 | batch=1,上下文 1024; FP16 基线与 W4A16 原生包均已运行 |

原始框架输入为 Token IDs、Attention Mask; 增量解码需 KV Cache.
输出为词表 Logits 与更新后的 KV Cache,最终由 Tokenizer 解码为文本.
FP16 基线静态 Decoder 输入依次为 hidden `[1,1,1536]`、past_key / past_value
`[1,2,1023,128]`、cosine / sine `[1,1,1,128]`、mask `[1,1,1,1024]`.
输出为 hidden_out `[1,1,1536]` 与当前 key / value `[1,2,1,128]`.
词表投影分为 10 片,前 9 片各 16384 行,最后一片 4480 行.
ONNX 为 FP32,板端 DLA 原生输入输出为 FP16,Runtime 加载时校验数量与字节大小.

## 预处理与解码

使用固定 revision 的 `tokenizer.json`、`tokenizer_config.json` 和
`generation_config.json`,保留官方 Chat Template 与特殊 Token.
上游建议采样 temperature=0.6,将任务要求写入 user 消息.
评测时固定随机种子与采样配置,记录停止条件、Token 数、截断情况及最终回答完成率.
中英文能力需要用本次固定样例与正式语料分别验证.

## 交付参考

地瓜 [RDK S 系列 Model Zoo](https://github.com/D-Robotics/rdk_model_zoo_s) 与
[S100 LLM 工具链](https://developer.d-robotics.cc/rdk_s_doc/Advanced_development/toolchain_development/LLM_Toolchain/s100_LLM_Toolchain)
仅作为模型选型、量化和指标展示参考.不使用厂商预编译 `.hbm` 作为 MTK 转换输入.

## 当前状态

官方资源、原始框架、标准导出和编译脚本已建立,完整 ONNX 一致性检查通过.
Genio 720 为 `board_verified`,运行 `20261010_deepseek_fp16_v3` 已完成中英文最终回答,
样例逐 Token NLL 数值检查通过; Genio 5100 为 `not_started`.
不将自编双语样例视为正式基准精度,也不将 FP16 结果视为 W4A16 验证.
环境核查范围、后续流程与限制见 [README](README.md).

原生 W4A16 使用 Prompt 128 Token、Decode 1 Token 的两个 DLA,INT16 embedding、KV Cache 和 I/O.
运行 `20261010_deepseek_w4a16_v2` 双语样例完成 EOS,Decode 约 16.75 Token/s.
英文样例存在语法问题,本次只确认板端推理,不作为正式量化质量验收.
