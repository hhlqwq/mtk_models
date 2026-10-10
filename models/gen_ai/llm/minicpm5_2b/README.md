# MiniCPM5-2B

OpenBMB 官方中英文纯文本 LLM,采用标准 Llama 架构.
实际参数量 2,516,756,480,42 层,hidden 2048,16 个 Query Head 和 2 个 KV Head.
官方模型: https://huggingface.co/openbmb/MiniCPM5-2B .
固定 revision: `f97400052a43d642bbc6e9975e2397e3ae6a6b52`,许可 Apache-2.0.
地瓜参考: https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s/samples/llm/minicpm5-2b .
地瓜产物只作为交付参考,不作为 MTK 转换输入.

当前处于适配中,尚未完成 MTK 编译、板端推理与正式质量评测.
初始目标 W4A16,Prefill 128,上下文 1024,Decode 1.
保留官方 RoPE 5000000、双停止 Token 1/130073 与独立 Chat Template,
默认 `enable_thinking=false`.只在独立转换副本中将新版 BPE merge 数组
序列化为旧版兼容字符串,官方资源保持原样.

在 Ubuntu89 仓库执行 `bash models/gen_ai/llm/minicpm5_2b/deploy/run_native.sh`,
资源使用当天临时目录中的 `minicpm5_source`,沿用已隔离的 GAI 2.7.5 环境.
可选阶段为 prepare、calibrate、quantize、shape、compile.
各阶段保存日志并遇错停止,编译成功不能代替板端推理验证.
打包入口为 `deploy/host/package_native.py`,要求输出为独立空目录.
官方资源地址、大小和哈希归档到 `models/source_manifest.json` 和 `models/source_url.txt`.
自编校准输入仅检查部署流程,正式量化质量需要独立的浮点/量化对比.

当前环境检查: GAI SDK 已接受官方 Llama 配置、42 层、RoPE 和双停止 Token.
通过独立目录安装 tokenizers 0.20.3 读取原始 tokenizer,与既有 0.19.1
读取兼容副本对比,4 个中英文、数字、特殊 Token 样例的 Token IDs 一致.
此检查只覆盖这些样例,不是完整 tokenizer 等价性证明.
独立参考 wheel 的 URL、大小及哈希已记录,未升级转换环境依赖.
官方权重 5033557096 字节已通过固定版本官方 SHA256 校验.
沿用的隔离环境为 `/tmp/hailongcodex/2026-10-10/deepseek_gai_env`,
新工作目录为 `/tmp/hailongcodex/2026-10-10/minicpm5_native`.

SDK 2.7.5 校准生成对 EOS 列表存在维度错误,独立转换配置使用官方对话结束 Token 130073.
原生交付 stopToken 仍保留 1 和 130073,官方原始配置保持不变.

主机质量对比入口为 `deploy/host/evaluate_quality.py`,同一固定 WikiText2 测试集
前 8192 Token,64 个 128 Token 块,每块重置上下文并评分 127 次下一 Token.
官方 Transformers FP32 与 SDK 量化 TFLite 使用相同 Token IDs,
该协议为测试集子集的转换质量检查,不作为完整基准或板端 NPU 精度结论.
原生 tokenizer 使用 RE2,数字切分配置为最多三位; RE2 不支持官方空白规则的
lookahead,默认使用官方 Token 输入.原生文本输入的兼容结论仅覆盖实际验证样例.

板端证据入口为 `deploy/board/evaluate_native.py`,默认运行中英文 Token/文本输入,
记录停止 Token、原始日志、Prefill/Decode 与本次内存观测,并核对两种输入结果.
日志需归档后由打包入口检查哈希; 推理成功不表示正式量化质量达标.

浮点质量入口同时生成相同双语 Prompt 的官方 FP32 Greedy 参考,
保留 Token 和实际文本以核对量化板端输出,不人工修饰差异.

官方 Safetensors 缺少旧 Transformers 加载器所需的 format 元数据,
浮点参考按官方索引直接加载全部 Tensor 并严格匹配模型参数,不重写原始权重.

Shape fixer 完成静态图后会复制真实 INT16 embedding.量化质量检查必须等该阶段完成,
入口仅校验与使用该实际资源并记录哈希,不得提前修改转换目录.

SDK 的 TFLite PPL 入口不接受 --dtype 参数,量化路径按实际静态图精度执行.

第一版基础 Hessian W4A16 的子集 PPL 为 148.45839,官方 FP32 为 39.55449,
相对增加 275.33%,未通过质量检查.同协议 SDK FP32 为 39.55450,
与官方参考一致.这版只保留转换证据,不能作为质量合格交付.

第二版使用独立工作目录 `minicpm5_native_v2`,校准混合 WikiText2 训练集
8 段各 256 Token 与 8 条自编中英文对话,不使用评测测试集进行校准.
`CALIBRATION_TRAIN` 指向已校验的训练集 Parquet;
`WEIGHT_OPT_CONFIG` 指向官方工具包的
`post_training_quantize/wgt_opt_cum_layer_error.json`,启用累计层误差优化
(512 个样本,batch size 1).质量与板端验证结果仍待实际运行.
可选 pytablewriter 表格依赖缺失时,评测入口仅在唯一完整 SDK JSON 已保存后
继续生成质量报告,其他异常仍直接报错.

校准成功后生成批次数完成标记,量化入口要求该标记匹配当前批次数,
防止校准未完成时启动量化.第二版提前启动的量化已作废,经确认停止后
保留日志与部分输出,等待 32 批校准完成再重新运行.
