# MODEL_NAME

来源、许可证和输入输出信息见 [模型卡](model_card.md).

## 第一步: 编译并上传

在 `deploy/run.sh` 顶部按五组填写模型与校准数据、产物与临时目录、板端地址与数据、ONNX 精度数据及编译环境.在编译主机的本模型目录运行:

```bash
bash deploy/run.sh
```

## 第二步: 开发板测试

登录配置的开发板,执行第一步打印的命令.进入部署目录后运行:

```bash
bash run.sh
```

## 数据与精度评测

填写本模型的数据目录格式、全量样本数、核心指标和 ONNX 评测依赖.来源记录放在 `models/source_url.txt`,量化方式在本 README 中说明.

## 当前测试结果

待上传 `results/summary.json` 后更新.记录 ONNX 核心精度、板端核心精度、精度变化、NPU 平均耗时和峰值 RSS.

## 板端部署结构

部署目录统一保留 `run.sh`、`board_paths.conf`、`models/`、`board/` 和测试结果.全量测试成功后保留 `BOARD_RESULTS_DIR/<运行编号>/summary.json` 及少量效果示例.

## 文件结构

`deploy/` 仅保留 `run.sh`,编译主机工具放在 `deploy/host/`,板端程序源码和评测代码放在 `deploy/board/`.模型产物放在 `models/`,示例按需保存在 `examples/`.

## 效果示例

选取少量固定输入放在 `examples/input/`,在编译时从配置的数据集生成来源及样本清单,放在构建目录并随部署上传,不纳入 Git.全量板端测试复用这些样本的真实预测,仅保存少量可读效果到本次结果目录的 `examples/output/`.取回本地后在此展示实际结果.
