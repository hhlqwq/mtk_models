# YOLOv8n / Genio 720

COCO 80 类目标检测,固定输入 `1x3x640x640` RGB,INT8 PTQ,目标为
MT8189 / MDLA 5.3.2026-10-08 已完成官方权重导出、INT8 量化、DLA 编译、
10 张冒烟和三后端 COCO val2017 全量评测,Genio 720 状态为完整交付.

## 与原有模型流程的核对

2026-10-08 复核现有入口,YOLOv5s、YOLO-World 和本模型均采用
"89 编译主机运行 deploy/run.sh,92 开发板运行上传的 run.sh"两步入口.

| 模型 | 转换与量化 | 板端执行 | 本次核对范围 |
| --- | --- | --- | --- |
| YOLOv5s | 官方源码加 MTK 补丁,TorchScript -> INT8 TFLite -> MDLA 5.3 DLA | C++ Neuron Runtime,COCO 5000 张 | 脚本与依赖路径复核,Shell 语法通过;保留原有实测结果 |
| YOLO-World XL | 现有 MediaTek 发布 ONNX 兼容性改写,无离线 INT8 量化 | ORT Neuron EP 板端编译;profiling 检查无 CPU fallback | 脚本和加速检查复核,Shell 语法通过;不预填独立 NPU 耗时 |
| YOLOv8n | 官方权重导出六个原始头,INT8 TFLite -> MDLA 5.3 DLA | C++ Neuron Runtime,CPU DFL 与 NMS | 本次执行导出、量化、编译及板端评测 |

FastSAM 的 Ultralytics 8.0.111 和原始头量化实现用于复用转换方式.
以上复核不会把旧模型的历史测试结果改记为本次重跑结果.
另已对 Depth Anything V2 Small、RTMPose、MobileFaceNet、ViT 和 Whisper
现有入口执行 Shell 语法检查,八个原有模型全部通过;本次没有修改这些模型实现.

## 官方来源

- 实现: https://github.com/ultralytics/ultralytics/tree/v8.0.111
- 安装版本: `ultralytics==8.0.111`,复用现有 Docker 包,不自动升级.
- 权重: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
- 权重 SHA256: `31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95`.
- 数据集: https://cocodataset.org/#download ,使用现有 COCO val2017.
- Qualcomm 仅作为交付结构参考,本模型不使用其预导出产物.

## 两步运行

先在 Windows 修改并提交代码,89 服务器拉取.将上述官方权重离线放入
`models/yolov8n.pt`,脚本会验证哈希,不会在线下载模型和依赖.

编译主机 `ubuntu89`,工作目录 `/data/users/hailong.he/github/mtk_models`:

```bash
bash models/perception/object_detection/yolov8n/deploy/run.sh
```

默认使用 Docker `hhl_g720_8011`,NeuroPilot SDK 8.0.11,MTK Converter 8.16.0,
100 张固定排序的 COCO 图片校准.脚本完成 ONNX 导出、INT8 TFLite 转换、
`ncc-tflite --arch=mdla5.3 --suppress-output --disallow-bridge` 编译,
PyTorch 与 ONNX 的 5000 张同协议精度评测、C++ 交叉编译和上传.
模型产物保存在模型的 `models/`,临时文件保存在
`/tmp/hailongcodex/当天日期/yolov8n`.

开发板 `root@192.168.0.92`:

```bash
bash /root/hailong.he/open_models/yolov8n/run.sh
```

板端使用已有 `/root/hailong.he/datasets/coco/val2017`,执行 20 次预热和
5000 张真实 NPU 推理,计算 mAP、平均 NPU 延迟、端到端延迟及进程峰值 RSS,
保存实际检测效果.结果位于部署目录 `results/运行ID/`.
不同运行使用独立目录,保留失败日志、预测、耗时和模型哈希,不自动删除.

## 适配与评测协议

三个尺度分别导出 64 通道框分布和 80 通道分类 logits,共六个输出.
DFL、Sigmoid、类别内 NMS 和原图坐标还原在 CPU 执行.
量化参数及输出索引从本次 TFLite 解析,通过 `runtime_config.csv` 部署;
Runtime 检查缓冲区尺寸,支持 NCHW 行宽按 16 对齐.
每次推理重新登记输入和全部输出,避免复用陈旧缓冲区.

PyTorch、ONNX 和板端使用相同 letterbox、单最佳类别、`conf=0.001`,
`IoU=0.6`,`max_det=300`.该协议用于本项目三后端对比,
不直接等同于 Ultralytics 官方公开指标的复现协议.
公开示例取自已有 COCO 数据,许可证和原始地址见
`examples/input/samples.json`,展示阈值为 0.25.

## 当前结果

运行 ID: `20261008_yolov8n_full_v1`,三个后端均使用 COCO val2017 全量
5000 张图片,主机与板端标注 SHA256 相同.

| 后端 | mAP@0.5:0.95 | 结果文件 |
| --- | ---: | --- |
| PyTorch FP32 | **36.6444%** | [pytorch_summary.json](results/pytorch_summary.json) |
| ONNX FP32 | **36.6260%** | [onnx_summary.json](results/onnx_summary.json) |
| MTK NPU INT8 | **35.3510%** | [summary.json](results/summary.json) |

ONNX 相对 PyTorch 变化为 **-0.0184 个百分点**,INT8 相对本次 ONNX
变化为 **-1.2750 个百分点**.单图原始头解码与官方前向的最大绝对误差为
`8.312659338116646e-5`,六个 ONNX 输出已通过数值一致性校验.

| 板端指标 | 平均 | P95 |
| --- | ---: | ---: |
| 预处理 | 13.236 ms | 16.963 ms |
| 独立 NPU 推理 | **6.283 ms** | 6.381 ms |
| CPU 后处理 | 3.233 ms | 4.669 ms |
| 端到端 | **22.925 ms** | **27.088 ms** |

推理进程峰值 RSS 为 **31.324 MiB**.NPU 耗时只统计常驻模型的
`NeuronRuntime_inference` 调用,排除 20 次预热;端到端从图片读取开始,
包含预处理、IO 登记、推理和后处理,不包含预测文件写入、COCO 汇总或画图.
完整分位数见 [timing_summary_current_run.json](results/timing_summary_current_run.json).

运行环境为 Rity Demo 26.0-release / scarthgap / Linux 6.6.137,
Neuron Runtime 8.2.16,CPU governor 为 `schedutil`;本次快照见
[environment.txt](results/environment.txt).主机使用 Torch 2.0.0+cu118,
ONNX 1.13.1,ONNX Runtime 1.18.0,ONNX 精度基准使用 CPU EP.
Genio 5100 尚未执行,上述数字只对应 Genio 720.

权重、ONNX、TFLite、DLA、张量契约、程序及代码哈希见
[delivery_manifest.json](results/delivery_manifest.json).
本次 DLA 大小为 `3,527,145` 字节,板端模型与编译主机产物哈希一致.
全部预测、图片覆盖清单、逐图耗时、日志和效果图保留在:

```text
/root/hailong.he/open_models/yolov8n/results/20261008_yolov8n_full_v1/
```

再次运行使用新的 `EVAL_RUN_ID` 或脚本生成的默认 ID,已有运行目录不会覆盖.

## 真实板端效果

展示阈值为 0.25,图像和许可证来源见 `examples/input/samples.json`.
效果由上述全量 NPU 预测生成,标签排布仅用于提高可读性.

![室内检测](examples/output/sample_1_detections.jpg)

![熊检测](examples/output/sample_2_detections.jpg)

![滑雪场景检测](examples/output/sample_3_detections.jpg)
