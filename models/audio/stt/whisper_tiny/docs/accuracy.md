# Whisper-Tiny 精度

LibriSpeech `test-clean` 同协议全量评测，2620/2620 条音频。主指标为 WER，越低越好。

| 后端 | WER |
| --- | ---: |
| OpenAI CUDA FP16 | 7.5546% |
| 开发板双 DLA | 7.5603% |

板端较参考端高约 0.0057 个百分点。原始结果见[板端报告](../results/full_accuracy/20260928_whisper_testclean_full_v1/summary.json)和[参考报告](../results/reference_accuracy/testclean_openai_cuda_v1/summary.json)。数据来源为 [OpenSLR SLR12](https://www.openslr.org/12/)。

历史 AISHELL-1 test 的主指标为 CER：开发板 45.5935%，OpenAI CUDA 45.8264%。该结果的绝对错误率较高，不作为 LibriSpeech 精度结论；详见 [历史报告](formal_eval_20260918_aishell1.json)。
