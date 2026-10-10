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
官方权重约 5.03 GB,当前下载完成前状态为 `weight_download_pending`.
沿用的隔离环境为 `/tmp/hailongcodex/2026-10-10/deepseek_gai_env`,
新工作目录为 `/tmp/hailongcodex/2026-10-10/minicpm5_native`.
