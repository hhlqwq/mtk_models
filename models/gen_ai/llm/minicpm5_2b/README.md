# MiniCPM5-2B

面向 Genio 720 的大语言模型适配,采用全 W4A16 量化,支持英文生成、性能统计及精度评价.
使用官方 [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B),许可证 Apache-2.0.
来源记录见 [source_url.txt](models/source_url.txt).

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

两端加 `-test` 进行少量评价.具体工具命令及参数由代码维护,运行时打印.

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
