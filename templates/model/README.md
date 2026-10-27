# MODEL_NAME

来源、许可证和输入输出见 [模型卡](model_card.md).本模板提供目录与入口骨架,需要实现模型业务后才能运行.

## 配置与执行

配置 NAS 模型与数据路径、转换参数和板端目录;公共环境从仓库平台配置读取.主机代码放在 `deploy/host/`,板端代码放在 `deploy/board/`,入口只编排步骤.

推理、前后处理、精度评价和性能统计统一使用 Python,通过 Python 调用 MTK SDK 动态库执行 NPU 推理.不新增 C++ 推理程序、自编桥接库或对应的 make 和交叉编译步骤.参考现有模型时,其尚未迁移的 C++ 实现不作为新模型规范.

进入编译容器后在本模型目录运行:

```bash
bash deploy/compile.sh
```

所有模型的主机 deploy/compile.sh 完成转换、量化、编译、打包和上传后结束,提示板端部署目录及命令,不通过 SSH 自动执行板端推理、示例生成或评价,不自动取回尚未生成的示例.上传完成只标为待板端检查.用户登录开发板,在部署目录执行 bash compile.sh 检查少量输入并保存 README 示例,不计算或发布指标.正式评价独立使用 evaluate.sh,只运行已有模型并统计精度、性能和进程峰值 RSS.每个模型 results/ 只保留 summary.json;板端预测、日志和中间指标统一放本模型部署目录的 work/,每次推理覆盖.新增模型需要实现示例生成、正式指标和覆盖校验,复用 tools/board.py 与 tools/summary.py.

入口直接调用命令,输出和错误直接显示在终端,只补充必要阶段提示.不默认增加日志文件或包装函数.复用 Docker 公共环境与公共路径工具,只保存运行、评分和排错必需的数据.

主流程独立步骤默认读取所属模型配置,不要求手填参数.普通容器步骤使用 python,板端使用 python3.所有模型统一通过 tools/board.py 使用部署目录内的 work/,不按日期或阶段建立过程目录;推理入口使用 context(__file__, new_run=True) 覆盖上次中间结果,评分和画图使用同一 work/.不要为单个模型增加临时目录规则.

用户登录开发板后,进入部署目录执行:

```bash
bash compile.sh
```

### 仅 LLM 的实时文本展示

适配本模板的 LLM 板端分支时,由 board/infer.py 生成并直接保存 examples/output 下的文本示例,移除 render_examples.py 的文件检查和调用.复用 tools/llm/generate.py 显示输入文本及实际聊天模板、Prefill 提示,增量刷新生成文本,正确处理未完成的 UTF-8 字符,结束时显示停止原因、生成 Token 数和示例位置.说明示例生成上限及是否仅为回答片段,不把 Token 进度或日志路径作为文本展示.

正式评价中的展示输入也可流式输出,独立预热和性能测量关闭文本打印,仅在测量外显示阶段提示.不将流式展示耗时作为独立性能测量,不逐题打印全量评分文本.上述要求仅针对 LLM,非 LLM 沿用原有示例生成流程.

## 数据与评价

`compile.sh` 使用固定少量输入计算 CosSim、RMSE、NRMSE,普通模型最多 5 次输入,不减少既有通路样本数.参考使用主机本地 NPZ,上传到板端 board/output_reference.npz,终端显示数值,不使用数据库或 NAS 评价张量.全量 evaluate.sh 只负责任务精度、性能与 RSS.复用 tools/output_metrics.py,实际接入输入、输出与版本核对,具体范围见[输出误差统计](../../docs/output_metrics.md).

填写数据目录格式、样本数、校准范围、评价协议及依赖.记录同协议原框架/ONNX 与 NPU 指标、差值、纯 NPU 和端到端耗时及峰值 RSS.多模型 DLA、外置 embedding 分项统计,文件大小采用十进制 MB.

原生 LLM 参考 MiniCPM5 或 DeepSeek 的双 DLA 流程,按本模型定义 tokenizer、停止 Token、Prefill/Decode 与 KV 接口.参考输入必须匹配已有部署图;完整题目超上下文时报告所需长度,不在评价入口截题或重新编译.

评价范围按模型明确配置: MiniCPM 为固定 1000 题、57 学科、seed=17 的 MMLU 分层子集及完整 WikiText2 test,参考资源用 evaluation/mmlu1000/;DeepSeek 保持完整 MMLU 与 WikiText2 test.不得把千题子集标为完整 MMLU 基准.参考计算读取实际设备配置,分别记录权重存储精度与参考计算精度;MMLU 校验提示词 Token 和单 Token 选项延续,不自动套用聊天模板.仅做 W4A16,掉点先比较原框架、SDK 浮点转换、W4A16 软件和 NPU,不默认追加实验或宣称损失合理.

## 结果与示例

`results/summary.json` 保存真实汇总,未完成评价不得填入示例数字.选取少量有来源和许可的输入,展示本次真实输出.全量数据和原始权重保存在 NAS,不进入 Git.

NAS 资源统一按 original/ 保存原始权重和来源,build/ 保存转换中间产物,最外层只放当前导出 ONNX 和最终 DLA;多文件部署包使用 models/.评价过程和缓存写主机本地日期临时目录,板端使用本模型 work/ 覆盖.
