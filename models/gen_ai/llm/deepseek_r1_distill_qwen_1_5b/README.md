# DeepSeek-R1-Distill-Qwen-1.5B

使用 [DeepSeek 官方模型](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).28 层、hidden=1536、Q/KV Head=12/2、RoPE=10000.详细版本、来源和许可见[模型卡](model_card.md)及[资源记录](models/source_url.txt).

## 执行范围

先在已有编译容器执行 `bash deploy/compile.sh`,完成转换、量化、编译与部署,随后自动调用板端 `compile.sh` 做少量通路检查并保存 README 示例到 `examples/output/`.该阶段不计算或发布精度、性能和峰值 RSS 报告.

正式评价使用独立的 `evaluate.sh`: 容器端只计算已有浮点模型的同协议参考及必要评价输入,资源单独放入已有部署包后,板端运行 `bash evaluate.sh`,统计精度、性能和推理进程峰值 RSS.正式结果只写入 `results/summary.json`,不重新量化或编译.预测、日志及内部指标位于当天日期临时目录.详见[板端规范](../../../../docs/genio_720.md).

下文已归档结果保留原始模型版本和评价范围,本次入口调整不代表重新实测.

DeepSeek 当前仅有固定生成样例,`evaluate.sh` 可报告样例性能与 RSS,正式全量精度仍标记未实现.下文唯一报告对应已有 W4A16 交付,不可当成 FP16 分片方案的新结果.

## 配置与运行

官方模型放在 NAS 对应模型资源目录的 `original/`,本模型配置见 [config.yaml](config.yaml).公共 Python、SDK、交叉编译器和开发板地址由编译容器加载.

在仓库根目录进入容器后执行 FP16 部署流程:

```bash
bash models/gen_ai/llm/deepseek_r1_distill_qwen_1_5b/deploy/compile.sh
```

入口完成导出、示例 Token 准备、转换、编译、桥接库构建和上传,自动在板端运行少量生成示例,不发布报告.FP16 示例文本解码需要已有 tokenizers 依赖.正式指标只在 evaluate.sh 中计算.

GAI 全 W4A16 转换使用独立入口:

```bash
bash models/gen_ai/llm/deepseek_r1_distill_qwen_1_5b/deploy/compile_native.sh
```

它依次执行 prepare、calibrate、quantize、shape 和 compile,不自动打包、上传或完成正式板端精度评价.默认量化为 `none`,不启用 Hessian;全部 Attention、MLP 和 LM Head 均为 `asym4W_sym16A`.打包运行 `python deploy/host/package_native.py`,默认输出到模型资源的 `models/`,该目录须为空.

## 输入与推理

上下文由 `conversion.context` 指定,生成上限由 `conversion.max_new_tokens` 指定.FP16 分片方案使用自有 Neuron Runtime 桥接库;GAI 方案使用 Prefill/Decode 双 DLA 和 `llm_cmdline_tool`.两套模型、embedding、tokenizer 及配置不能混用.

校准使用训练集或自编对话,正式测试集不参与校准.如扩展训练校准样本,使用 `tools/llm/prepare_calibration.py`,固定采样位置和旋转基,并重新生成校准数据.

## 评价与状态

已有 FP16 和 W4A16 均完成固定样例生成,不能据此认定正式质量合格.W4A16 的生成及性能证据见 [统一报告](results/summary.json),新配方的 NPU MMLU/WikiText 正式对比仍未完成.


MMLU 使用同一题目 Token 和空格加 A/B/C/D 的单 Token logits 评分,不加 Chat Template、不截断或丢题.WikiText 以 128 Token 分块、每块重置上下文,评分 127 次下一 Token.当前全量入口尚未接入正式协议;扩展评价时若完整题目超出已有模型上下文,应停止并另行规划部署调整,不截题或在评价中重编译.

精度检查依次比较原框架 FP32、SDK 浮点、量化 TFLite 和真实 MDLA 输出,分别定位结构转换、量化及板端执行误差.质量状态以实际评价结果为准,不通过提高权重位宽掩盖 W4A16 退化.
