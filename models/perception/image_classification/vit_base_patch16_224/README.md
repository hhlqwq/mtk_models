# ViT-Base Patch16 224

完整测试成功后只保留一个 `summary.json`,汇总板端 NPU 平均耗时、核心精度及参考基准的精度差值。原始预测、缓存与日志在汇总成功后自动删除; 失败时保留本次 `work/` 目录。`REFERENCE_ACCURACY` 与 `REFERENCE_SOURCE` 在脚本顶部配置,历史参考会明确标注; 缺少匹配基准时不计算差值。

运行时的临时文件与 C++ 程序保存在脚本顶部的 `BUILD_WORK_DIR`,默认是仓库外的 `/tmp/hailongcodex/<当天日期>/vit_base_patch16_224/`。模型产物写入 `MODEL_OUTPUT_DIR`,板端测试结果写入 `BOARD_RESULTS_DIR`。脚本不在代码目录生成 Python 字节码缓存。

ImageNet-1K 图像分类，输入为 224×224 RGB 图像。正式上游为 PyTorch Vision v0.15.1，来源见[模型卡](model_card.md)。

## 运行

1. 在 [run.sh](deploy/run.sh) 中配置 ONNX、校准图片、ImageNet 验证集、模型输出目录、Docker 与交叉编译环境、板端地址和部署目录。
2. 在编译主机的本模型目录执行 `bash deploy/run.sh`。脚本在 Docker 中量化并编译 DLA，在主机交叉编译板端 C++ 程序并上传。
3. 登录所配置的开发板，执行脚本打印的板端命令。板端处理 50000 张验证图片并计算 Top-1；结果写入 `BOARD_RESULTS_DIR`。

## 历史精度

正式开源模型在 ILSVRC2012 val 全量 50000 张图片上完成同协议评测，主指标为 Top-1。

| 后端 | Top-1 |
| --- | ---: |
| FP32 ONNX | 80.64% |
| MTK NPU INT8 | 79.38% |

板端较 ONNX 低 1.26 个百分点。历史 Qualcomm 预导出模型的指标不计入这组结果。修改后的脚本尚未重新实测。

## 历史性能

正式开源权重衍生 DLA 在 Genio 720 EVK 上连续推理 100 次，纯 NPU 平均延迟为 **51.7129 ms/次**，峰值 RSS 为 94504 KB。纯 NPU 延迟不含图片解码、resize、中心裁剪、输入量化或后处理；单次进程端到端测得 171 ms，包含进程启动和模型加载。这些是历史测量值，不能当作修改后脚本的验证结果。
