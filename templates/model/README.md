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

## 当前测试结果

待上传 `results/summary.json` 后更新.记录 ONNX 核心精度、板端核心精度、精度变化、NPU 平均耗时和峰值 RSS.

## 文件结构

`deploy/` 仅保留 `run.sh`,Python 和 C++ 源码分别放在 `deploy/python/` 和 `deploy/cpp/`.模型产物放在 `models/`,示例按需保存在 `examples/`.
