# MiniCPM5-2B

面向 Genio 720 / MDLA 5.3 的全 W4A16 适配.使用官方 [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) 模型,许可证 Apache-2.0.42 层、hidden=2048、Q/KV Head=16/2,约 2.52B 参数.详细版本与文件校验信息见[来源清单](models/source_manifest.json).其他 Model Zoo 仅作为交付结构参考.

## 一键执行

参数统一放在本模型的 `config.yaml`,工具链环境由仓库 `docker/enter_container.sh` 加载.正常执行不传阶段参数,只提供 `-test` 少量评价选项.

主机步骤统一使用当前终端的 `python`,校准子进程沿用启动脚本的解释器.在本模型目录中执行转换:

```bash
python deploy/host/convert.py
```

默认 `native.calibration_train` 为空,使用自编校准样例,转换代码不导入 `pyarrow`.配置 Parquet 训练集后才加载 `pyarrow`;MMLU/WikiText2 评价也需要它来读取数据.

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

脚本依次调用模型转换、DLA 编译、FP32 参考评价、桥接库构建和打包上传五个步骤.每步显示进度,外部工具的实际命令由对应实现打印,执行错误立即停止.入口不检查环境,不安排资源或数据预检,不会自动在板端执行推理.

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
| 生成检查 | 英文执行 Token/文本输入,要求正常停止 | 同样两次生成,最多 32 Token,不要求上限内停止 |
| 统一性能 | 128 Token 输入,最多生成 128 Token | 相同定长协议 |
| 校准与量化 | 按 `config.yaml` 执行 | 相同配方,不以缩小校准换取快速评价 |
| 结果标记 | `evaluation_scope=full` | `evaluation_scope=subset,status=smoke_test` |
| 最新汇总 | `results/summary.json` | `results/summary_test.json` |

默认完整部署包同时携带全量和快速参考,板端可直接选择 `run.sh` 或 `run.sh -test`.如果先在容器运行 `-test`,该部署包只携带快速评价输入;板端直接运行无参数入口会明确拒绝,需要先准备并上传全量包.快速结果不覆盖全量汇总.默认小子集数量可在 `evaluation` 配置中调整.

正常生成但未在上限内停止,或文本/Token 输入结果不同,会记录为生成检查未通过,继续执行独立的精度评价.最终汇总保留全部已完成指标,状态为 `generation_checks_failed`,入口返回非零退出码.Runtime 执行失败或日志缺少必要计数时立即停止.全量与快速评价均不把失败标为成功.

## 执行顺序与代码位置

先读 `deploy/run.sh`,写法与 YOLOv5s、ViT、MobileFaceNet 的入口一致:解析 `-test`、区分主机和板端,然后顺序调用实际步骤.入口不定义函数,不检查环境或数据:

| 顺序 | 实际工作 | 实现位置 |
|---|---|---|
| 主机 1 | 模型准备、校准、量化、静态图导出 | `deploy/host/convert.py` |
| 主机 2 | Prefill/Decode 编译 | `deploy/host/compile.py`,调用 `ncc-tflite` 并打印完整命令 |
| 主机 3 | FP32 参考与板端输入 | `deploy/host/evaluate.py` |
| 主机 4 | 桥接库构建 | `make`,规则在 `deploy/board/Makefile` |
| 主机 5 | 打包上传 | `deploy/host/upload.py`,直接调用 `package.py` 的打包函数,然后执行 `ssh`、`rsync` |
| 板端 1 | 生成、输入一致性、定长性能 | `deploy/board/infer.py`,直接执行 `llm_cmdline_tool` |
| 板端 2 | MMLU、WikiText2 | `deploy/board/evaluate.py`,直接调用公共评价函数 |
| 板端 3 | 协议核对和结果汇总 | `deploy/board/summarize.py` |

`convert.py` 直接导入 `tools/llm/quantize_native.py` 的 `quantize()` 和 `fix_native_shapes()`,后者保留 SDK 异常传播、W4A16 核对及 A16 接口适配.量化和导出不再启动仓库内的另一个 Python 脚本.校准沿用现有 SDK 命令入口,直接执行 当前 Python 解释器同目录的 `mtk_make_llm_ptq_calib_dataset`,完整命令在终端打印.SDK 的 `main()` 会自行解析参数,参数转换只保留在 SDK 接口处.

## 外部工具的具体命令

以下用于理解或手动排查单个步骤.编译和推理实现会打印替换为实际路径的完整命令,包括输入文件和参数;板端还会打印工作目录.手动执行不生成完整评价汇总,不要与同模型的一键任务并发执行.

### Docker 内编译 Prefill 和 Decode

先完成 `convert.py` 的转换,在容器仓库根目录可直接运行编译步骤.它读取模型配置,自动定位本次两张静态图,按下面的完整配方编译:

```bash
python models/gen_ai/llm/minicpm5_2b/deploy/host/compile.py
```

如果需要绕过 Python 直接调用编译器,先将转换日志中 `[Prefill 静态图]` 和 `[Decode 静态图]` 后面的完整路径分别填入以下变量.这是本次 NAS 转换产物的路径,不是仓库代码路径.编译器位置 `NCC_ROOT` 由 `docker/enter_container.sh` 加载的本地平台配置提供.

```bash
PREFILL_TFLITE='/替换为转换日志中的Prefill静态图绝对路径.tflite'
DECODE_TFLITE='/替换为转换日志中的Decode静态图绝对路径.tflite'
```

Prefill 的实际编译命令:

```bash
env LD_LIBRARY_PATH="${NCC_ROOT}/lib" "${NCC_ROOT}/bin/ncc-tflite" \
    --arch=mdla5.3 --l1-size-kb=256 --num-mdla=1 \
    --opt=3 --opt-footprint --opt-accuracy --stable-linearize \
    --gno=LTS,Inception --gno-exp --gno-non-4d-tiling --mlo \
    --disable-apusys --fc-to-conv --mdla-int16-lut \
    --suppress-input --suppress-output --show-memory-summary \
    --mdla-conv-exp --intval-color-legacy --disallow-bridge \
    --broadcast-act-wgt --broadcast-flow-distance=63 --split-large-conv-ic=1536 \
    "${PREFILL_TFLITE}" --dla-file "${PREFILL_TFLITE%.tflite}.dla"
```

Decode 的实际编译命令:

```bash
env LD_LIBRARY_PATH="${NCC_ROOT}/lib" "${NCC_ROOT}/bin/ncc-tflite" \
    --arch=mdla5.3 --l1-size-kb=256 --num-mdla=1 \
    --opt=3 --opt-footprint --opt-accuracy --stable-linearize \
    --gno=LTS,Inception --gno-exp --gno-non-4d-tiling --mlo \
    --disable-apusys --fc-to-conv --mdla-int16-lut \
    --suppress-input --suppress-output --show-memory-summary \
    --mdla-conv-exp --intval-color-legacy --disallow-bridge \
    --split-16a4w-conv-oc \
    "${DECODE_TFLITE}" --dla-file "${DECODE_TFLITE%.tflite}.dla"
```

两条命令保持原有编译配方.`--arch=mdla5.3` 指定目标架构,`--disallow-bridge` 禁止 CPU 桥接,`--suppress-input`/`--suppress-output` 保留与运行代码配套的原生整数接口.两阶段的区别是 Prefill 使用三项 broadcast/large-conv 参数,Decode 使用 `--split-16a4w-conv-oc`.输出 DLA 位于对应 TFLite 的同目录.

### 开发板直接生成

先进入上传时打印的板端部署目录,以下相对路径均以该目录为工作目录.英文官方 Token 输入,总生成上限 512 Token:

```bash
llm_cmdline_tool scripts/config-yocto.yaml --read-tokens \
    -i scripts/prompts/en_demo.tokens.txt -m 511
```

同一英文 Prompt 的文本输入,使用 CLI 内置分词器:

```bash
llm_cmdline_tool scripts/config-yocto.yaml \
    -i scripts/prompts/en_demo.txt -m 511
```

这里的文本文件已经包含官方聊天模板,不要直接把不带模板的裸问题当成相同输入.生成检查只保留英文,每次执行英文 Token 输入、英文文本输入和固定性能共三次 CLI 调用.校准样本属于量化配方,与生成测试分开管理,本次没有修改.

固定性能命令,输入 128 Token,总生成上限 128 Token:

```bash
llm_cmdline_tool scripts/config-yocto.yaml --read-tokens \
    -i scripts/prompts/performance_128.tokens.txt -m 127
```

`--read-tokens` 表示输入文件是空格分隔的 Token ID;不加时输入为文本.`-m` 是首 Token 之后的生成调用上限,因此总上限 512/128/32 分别使用 511/127/31,遇到停止 Token 可以提前结束.无参数一键流程的生成上限来自 `conversion.max_new_tokens`,上面的 512 是当前默认值.

`scripts/config-yocto.yaml` 指定配套的两张 DLA、embedding、分词文件、停止 Token 和缓存形状.手动命令与 `infer.py` 使用同一份 YAML.板端 MMLU/PPL 另由 Python 通过 Neuron 桥接库执行,不属于 `llm_cmdline_tool` 的这几条生成命令.

## 运行前准备

1. 在仓库 `configs/storage.local.json` 配置 `models_root` 和 `dataset_root`,确认容器挂载及 SDK、GAI Python、交叉编译环境已经准备好.
2. 官方模型及 `source_manifest.json` 放在外部资源目录 `models_root/gen_ai/llm/minicpm5_2b/original/`.具体文件、下载 URL、大小和 SHA256 见 [来源记录](models/source_url.txt) 与 [来源清单](models/source_manifest.json).入口离线运行,不自动下载.
3. 数据集统一放在 NAS `dataset_root` 内,`config.yaml` 中的 `data.wikitext_test` 和 `data.mmlu_test` 按该根目录定位.默认使用 `wikitext/wikitext-2-raw-v1/` 和 `mmlu/all/`.执行步骤直接读取配置路径,不提前扫描数据、不搜索临时目录或自动下载.
4. WikiText2 使用 [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) 的 `wikitext-2-raw-v1`;MMLU 使用 [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) 的 test 集.校准只能使用训练数据或独立自编样例,测试集不参与校准.

**完整 MMLU 与当前交付的 1024 Token 上下文不是同一个已验证范围.** 不再执行量化前的题集长度预检;评价步骤实际处理题目时会记录输入长度,超出上下文则报错,不截断或丢题.更大上下文需要重新转换和编译,其内存和板端兼容性尚未验证.

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
│   ├── host/
│   │   ├── convert.py       准备、校准、量化和静态图导出
│   │   ├── compile.py       编译 Prefill 和 Decode,打印完整 NCC 命令
│   │   ├── evaluate.py      FP32 参考评价和板端输入生成
│   │   ├── package.py       上传使用的部署包构建函数
│   │   └── upload.py        打包并上传
│   └── board/
│       ├── Makefile         构建共享桥接库
│       ├── infer.py         直接调用 CLI 并采集生成和性能记录
│       ├── evaluate.py      NPU 精度评价
│       └── summarize.py     汇总和状态判断
└── results/
    ├── summary.json          当前已有交付模型成绩
    ├── evidence.json         首轮评价、性能、短生成和质量证据
```

`tools/llm/` 只复用 SDK 转换适配及评价算法,模型的执行顺序和 CLI 生成逻辑分别在本目录的 `run.sh` 和 `board/infer.py`.公共 Neuron 桥接源码为 `common/cpp/neuron_bridge.cpp`,由 Makefile 直接构建.PPL/MMLU 通过桥接库调用真实 Neuron MDLA Runtime,不回退 CPU.板端需要 Python、NumPy 和匹配的 Runtime,不需要主机的 Torch、Transformers 或 Converter.

`host/` 只放可独立执行的步骤文件.模型准备函数与转换逻辑合并在 `convert.py` 中,打包由 `upload.py` 直接调用 `package.py` 的函数,不额外启动 Python 子进程.编译参数和执行在 `compile.py`,板端评价与汇总在 `deploy/board/`.`run.sh` 只顺序调用步骤,不定义辅助函数或编译函数,也不通过额外的环境文件传递模型路径.

在 Docker 的仓库根目录可依次手动执行:

```bash
python models/gen_ai/llm/minicpm5_2b/deploy/host/convert.py
python models/gen_ai/llm/minicpm5_2b/deploy/host/compile.py
python models/gen_ai/llm/minicpm5_2b/deploy/host/evaluate.py
make -C models/gen_ai/llm/minicpm5_2b/deploy/board
python models/gen_ai/llm/minicpm5_2b/deploy/host/upload.py
```

`upload.py` 包含打包步骤.如果只生成部署包,最后一步改为执行 `host/package.py` 即可.手动快速评价在整组命令前设置 `export MTK_EVAL_TEST=1`,默认全量使用 `export MTK_EVAL_TEST=0`.

## 资源和部署目录

外部模型资源根目录为 `models_root/gen_ai/llm/minicpm5_2b/`,只保留 `original/`、`models/` 和 `build/` 三类目录.官方权重及来源记录在 `original/`,当前部署包在 `models/`,本次转换使用 `build/native/`,打包暂存使用 `build/package/`.新转换替换旧工作目录;新包完整生成后替换当前部署包.上传成功后保留当前转换资源供板端诊断,下一次转换替换,不积累历史副本.上传成功只表示传输完成,不代表板端验证通过.

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

同一模型重复执行替换对应模式的上一次输出,运行编号保存在结果内容中.入口和传输使用同一板端文件锁拒绝并发操作.上传同步替换旧部署文件和结果,不扫描进程或逐文件复核哈希;生成前不执行整包扫描.配置快照保存实际转换参数,包括环境变量覆盖的上下文和量化参数.

仓库和 NAS 不保存淘汰模型、旧部署包、历史诊断及逐次实验记录.只保留当前选定模型的来源、最新汇总和必要评价证据.已有包尚未重新打包上传,不能视为已经使用本次英文入口.新包统一使用根目录 `run.sh`;直接生成使用上面的英文 CLI 命令.

## 当前结果

以下为当前交付模型的固定首轮子集成绩,不是全量基准.全 W4A16,上下文 1024 Token,Prefill 每批 128 Token,Decode 每步处理一个 Token.精度变化如实列出,没有明确验收阈值时不预设通过或不通过.

| 指标 | 官方 FP32 | 当前交付 W4A16 |
|---|---:|---:|
| MMLU,固定 1000 题 | 55.9% | 41.3% |
| WikiText2,前 128 行 | 40.0411 | 48.3821 |

统一性能为 Prefill 288.67 token/s、Decode 8.15 token/s,测速输入长度为 128 Token、生成上限为 128 Token,这不是模型的最大输入或输出长度.该次性能进程峰值 RSS 为 1152.42 MiB.短生成四题均包含正确答案并停止,严格按要求只输出答案为 3/4.

全量和快速评价的 `summary.json` 同时汇总生成 CLI 和 MMLU/WikiText 评价 Python 进程的峰值 RSS,记录各阶段值和其中最大值.当前已有包的 Demo 入口也在 `results/latest_demo/summary.json` 记录生成进程峰值 RSS.这些值是最大单进程内存,不代表设备总内存或全部 NPU 驱动内存.

[统一汇总](results/summary.json)、[评价证据](results/evidence.json) 和 [适配说明](ADAPTATION.md) 记录当前选定模型的范围和限制.新入口只有完成对应板端评价、核对输入及模型哈希后才产生新汇总,运行完成也不自动宣布精度合格.
