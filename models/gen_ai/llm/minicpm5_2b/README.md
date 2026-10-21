# MiniCPM5-2B

面向 Genio 720 的大语言模型适配,采用全 W4A16 量化,支持英文生成、性能统计及精度评价.
使用官方 [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B),许可证 Apache-2.0.
来源记录见 [source_url.txt](models/source_url.txt).

## 执行范围

`deploy/run.sh` 和板端 `run.sh` 无参数默认小批量验证: 配置指定的 MMLU 16 题和 WikiText2 4 块及功能示例.评价范围不改变校准、量化或编译配方.

全量精度使用独立入口: 容器执行 `bash deploy/evaluate.sh` 生成已有浮点模型的全量参考,单独放入现有部署包后,板端执行 `bash evaluate.sh`.该流程不量化、不编译、不上传,仅推理和计算精度,结果保存为本次运行的 `accuracy_summary.json`.详见[板端评价规范](../../../../docs/genio_720.md).

下文已归档指标保留原运行范围,不因默认入口变化而更新.

## 运行

模型和数据路径、转换及评价参数统一在 [config.yaml](config.yaml) 配置.
模型资源和数据集提前放在 NAS,运行时不自动下载.

进入编译容器后,在本模型目录执行转换、编译、参考评价及上传:

```bash
bash deploy/run.sh
```

在开发板部署目录执行生成、性能和精度评价:

```bash
bash run.sh
```

两端无参数默认少量评价,工具命令和参数由执行代码维护并打印.

## 当前结果

以下为已有交付模型的固定子集成绩,不是全量基准.

| 指标 | 官方 FP32 | W4A16 |
|---|---:|---:|
| MMLU,1000 题 | 55.9% | 41.3% |
| WikiText2 PPL,前 128 行 | 40.04 | 48.38 |
| Prefill 速度 | — | 288.67 Token/s |
| Decode 速度 | — | 8.15 Token/s |
| 推理进程峰值 RSS | — | 1152.42 MiB |

以上成绩对应已归档的量化配方,不代表默认配置重新转换后的结果.
完整结果及适配说明见 [summary.json](results/summary.json)、[evidence.json](results/evidence.json) 和 [ADAPTATION.md](ADAPTATION.md).
