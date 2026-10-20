# MiniCPM5-2B

面向 Genio 720 / MDLA 5.3 的全 W4A16 适配.使用官方 [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) 模型,许可证 Apache-2.0.42 层、hidden=2048、Q/KV Head=16/2,约 2.52B 参数.详细版本与文件校验信息见[来源清单](models/source_manifest.json).其他 Model Zoo 仅作为交付结构参考.

## 一键执行

参数统一放在本模型的 `config.yaml`,工具链环境由仓库 `docker/enter_container.sh` 加载.正常执行不传阶段参数,只提供 `-test` 少量评价选项.

在编译主机的仓库根目录进入 Docker:

```bash
bash docker/enter_container.sh
```

进入容器后,在仓库根目录执行完整转换、参考评价、打包及上传:

```bash
bash models/gen_ai/llm/minicpm5_2b/deploy/run.sh
```

少量评价模式:

```bash
bash models/gen_ai/llm/minicpm5_2b/deploy/run.sh -test
```

脚本先检查 NAS 数据集和模型路径,再依次执行官方资源校验、题集长度预检、校准、量化、静态图导出、双 DLA 编译、FP32 参考评价、公共桥接库交叉编译、打包和上传.每步显示进度,失败立即停止.上传后逐文件核对 SHA256,不会自动在板端执行推理.

上传后,在开发板进入脚本输出的部署目录,执行:

```bash
bash run.sh
```

板端自动执行生成及文本/Token 输入一致性检查、统一性能、MMLU、WikiText2 PPL 和指标汇总.少量评价使用:

```bash
bash run.sh -test
```

| 项目 | 默认 `run.sh` | `run.sh -test` |
|---|---|---|
| MMLU | 完整 test,14,042 题 | seed=17 固定 16 题 |
| WikiText2 | 完整 test 文本,128 Token 分块 | 前 128 行中前 4 块,共 508 个评分位置 |
| 生成检查 | 中英文各执行 Token/文本输入,要求正常停止 | 同样四次生成,最多 32 Token,不要求上限内停止 |
| 统一性能 | 128 Token 输入,最多生成 128 Token | 相同定长协议 |
| 校准与量化 | 按 `config.yaml` 执行 | 相同配方,不以缩小校准换取快速评价 |
| 结果标记 | `evaluation_scope=full` | `evaluation_scope=subset,status=smoke_test` |
| 最新汇总 | `results/summary.json` | `results/summary_test.json` |

默认完整部署包同时携带全量和快速参考,板端可直接选择 `run.sh` 或 `run.sh -test`.如果先在容器运行 `-test`,该部署包只携带快速评价输入;板端直接运行无参数入口会明确拒绝,需要先准备并上传全量包.快速结果不覆盖全量汇总.默认小子集数量可在 `evaluation` 配置中调整.

## 运行前准备

1. 在仓库 `configs/storage.local.json` 配置 `models_root` 和 `dataset_root`,确认容器挂载及 SDK、GAI Python、交叉编译环境已经准备好.
2. 官方模型及 `source_manifest.json` 放在外部资源目录 `models_root/gen_ai/llm/minicpm5_2b/original/`.具体文件、下载 URL、大小和 SHA256 见 [来源记录](models/source_url.txt) 与 [来源清单](models/source_manifest.json).入口离线运行,不自动下载.
3. 数据集统一放在 NAS `dataset_root` 内,`config.yaml` 中的 `data.wikitext_test` 和 `data.mmlu_test` 按该根目录定位.默认使用 `wikitext/wikitext-2-raw-v1/` 和 `mmlu/all/`,禁止指向 `/tmp`.入口在读取权重前检查文件和版本,不会搜索临时目录或自动下载.
4. WikiText2 使用 [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) 的 `wikitext-2-raw-v1`;MMLU 使用 [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) 的 test 集.校准只能使用训练数据或独立自编样例,测试集不参与校准.

**完整 MMLU 与当前交付的 1024 Token 上下文不是同一个已验证范围.** 入口在校准、量化前检查所选题集的最长输入,将长度信息写入本次 `evaluation/运行编号/模式/mmlu/length_report.json`.如果超出 `conversion.context`,会报告所需长度并停止.应将上下文调整到足够大的 128 整数倍后重新转换和编译;不截断、不丢题、不把固定千题冒充全量.更大上下文的编译、内存和板端兼容性尚未验证.

默认量化配置为常规 W4A16、32 批校准、未启用旋转.已有交付模型使用旋转 seed=0、128 批校准及归档的 16 样本分阶段 Hessian 配方,直接运行默认配置不会复现同一模型.

## 源码结构

```text
minicpm5_2b/
├── README.md                 使用方法、评价范围与当前指标
├── ADAPTATION.md             转换约定与精度检查
├── config.yaml               模型、数据集、转换和评价配置
├── models/                   官方来源 URL 与哈希清单
├── deploy/
│   ├── run.sh                主机/板端共用的一键入口
│   ├── host/                 准备、校准、量化、导出、编译、参考评价、打包、上传
│   └── board/                生成、NPU 评价、汇总入口及 Makefile
└── results/
    ├── summary.json          当前已有交付模型成绩
    ├── evidence.json         首轮评价、性能、短生成和质量证据
```

公共 Python 实现在仓库 `tools/llm/`;公共 Neuron 桥接源码为 `common/cpp/neuron_bridge.cpp`,MiniCPM 和 DeepSeek 的 Makefile 共用.板端生成调用系统 `llm_cmdline_tool`,PPL/MMLU 通过桥接库调用真实 Neuron MDLA Runtime,不回退 CPU.板端需要 Python、NumPy 和匹配的 Runtime,不需要主机的 Torch、Transformers 或 Converter.

`deploy/host/package_model.py` 只负责构建部署包;板端评价与汇总在 `deploy/board/` 中完成,不再由打包脚本接收结果并更新状态.内部 Python 文件仍然独立,便于定位失败步骤,无需人工逐个执行.

## 资源和部署目录

外部模型资源根目录为 `models_root/gen_ai/llm/minicpm5_2b/`,只保留 `original/`、`models/` 和 `build/` 三类目录.官方权重及来源记录在 `original/`,当前部署包在 `models/`,本次转换使用 `build/native/`,打包暂存使用 `build/package/`.新转换替换旧工作目录;新包完整生成后替换当前部署包,上传并核对成功后删除转换中间文件.失败保留本次现场,下次运行替换,不积累历史副本.

上传后的板端目录为:

```text
minicpm5_2b/
├── run.sh                    与主机相同的入口
├── config.json               已解析的模型配置及准备模式
├── manifest.json             交付文件清单、精度及图接口
├── 1024c/                    示例形状,目录名随实际上下文变化
│   ├── prompt.dla
│   └── decode.dla
├── tokenizer/                INT16 embedding 和分词资源
├── scripts/                  原生 CLI YAML 及 Prompt
├── board/                    模型执行代码、公共模块及 AArch64 桥接库
├── evaluation/full或test/    与本次模型匹配的题集、输入契约和 FP32 参考
└── results/                 最新全量和快速结果,分别位于 latest/ 和 latest_test/
```

同一模型重复执行替换对应模式的上一次输出,运行编号保存在结果内容中.入口使用文件锁拒绝并发运行.上传同步删除过期部署文件,保留最新结果;发现进程占用部署目录时拒绝覆盖,不停止其他进程.

仓库和 NAS 不保存淘汰模型、旧部署包、历史诊断及逐次实验记录.只保留当前选定模型的来源、最新汇总和必要评价证据.当前已有包仍可使用 `bash scripts/run.sh` 执行双语 Demo,仅保留 `results/latest_demo/`;该入口不代表全量评价.

## 当前结果

以下为当前交付模型的固定首轮子集成绩,不是全量基准.全 W4A16,Prefill=128、Decode=1、上下文=1024,`quality_qualified=false`.

| 指标 | 官方 FP32 | 当前交付 W4A16 |
|---|---:|---:|
| MMLU,固定 1000 题 | 55.9% | 41.3% |
| WikiText2,前 128 行 | 40.0411 | 48.3821 |

统一性能为 Prefill 288.67 token/s、Decode 8.15 token/s,使用固定 128 Token 输入、最多生成 128 Token.性能运行未到 EOS,不代表自由生成质量通过.短生成四题均包含正确答案并停止,严格按要求只输出答案为 3/4,不能表述为指令遵从全部通过.

[统一汇总](results/summary.json)、[评价证据](results/evidence.json) 和 [适配说明](ADAPTATION.md) 记录当前选定模型的范围和限制.新入口只有完成对应板端评价、核对输入及模型哈希后才产生新汇总,运行完成也不自动宣布精度合格.
