# MiniCPM5-2B

> Python 推理代码已迁移,尚未做板端验证.下文报告和示例属于迁移前的 C++/原生 CLI 实现,不能作为当前 Python 实现的精度、性能或 RSS.

面向 Genio 720 的大语言模型适配,采用全 W4A16 量化,支持英文生成、性能统计及精度评价.
使用官方 [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B),许可证 Apache-2.0.
来源记录见 [source_url.txt](models/source_url.txt).

## 执行范围

先在已有编译容器执行 `bash deploy/compile.sh`,完成转换、量化、编译与部署,随后自动调用板端 `compile.sh` 做少量通路检查并保存 README 示例到 `examples/output/`.该阶段不计算或发布精度、性能和峰值 RSS 报告.

正式评价使用独立的 `evaluate.sh`: 容器端只计算已有浮点模型的同协议参考及必要评价输入,资源单独放入已有部署包后,板端运行 `bash evaluate.sh`,统计精度、性能和推理进程峰值 RSS.正式结果只写入 `results/summary.json`,不重新量化或编译.预测、日志及内部指标位于当天日期临时目录.详见[板端规范](../../../../docs/genio_720.md).

下文已归档结果保留原始模型版本和评价范围,本次入口调整不代表重新实测.

## 运行

模型和数据路径、转换及评价参数统一在 [config.yaml](config.yaml) 配置.
模型资源和数据集提前放在 NAS,运行时不自动下载.

进入编译容器后,在本模型目录执行转换、编译及上传,自动生成板端示例:

```bash
bash deploy/compile.sh
```

正式评价前按[板端规范](../../../../docs/genio_720.md)准备完整 MMLU/WikiText2 输入和参考,再在开发板部署目录执行:

```bash
bash evaluate.sh
```

compile.sh 仅生成少量示例,正式指标统一由 evaluate.sh 给出.

## 迁移前结果

以下为已有交付模型的固定子集成绩,不是全量基准.

| 指标 | 官方 FP32 | W4A16 |
|---|---:|---:|
| MMLU,1000 题 | 55.9% | 41.3% |
| WikiText2 PPL,前 128 行 | 40.04 | 48.38 |
| Prefill 速度 | — | 288.67 Token/s |
| Decode 速度 | — | 8.15 Token/s |
| 推理进程峰值 RSS | — | 1152.42 MiB |

以上成绩对应已归档的量化配方,不代表默认配置重新转换后的结果.
结果及适配说明见 [summary.json](results/summary.json) 和 [ADAPTATION.md](ADAPTATION.md).
