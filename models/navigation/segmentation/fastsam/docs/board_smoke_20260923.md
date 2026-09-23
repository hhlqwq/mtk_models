# FastSAM-s Genio 720 C++ 板端冒烟,2026-09-23

## 运行结论

官方 FastSAM-s 权重完成 PyTorch 前向、opset 13 ONNX 导出、MTK INT8 量化、
MDLA 5.3 编译.92 开发板的 C++ 程序通过 Neuron Runtime API 加载 DLA,
对一张 1280×853 图片执行真实硬件推理,生成 100 个实例掩码和叠加图.
当前状态为 `board_verified`,尚不具备正式精度或稳定性能结论.

## 来源与环境

| 项目 | 本次实际值 |
| --- | --- |
| 模型权重 | 官方 README 链接的 FastSAM-s.pt,用户本机下载并同步至 89 |
| 权重大小 | 23,832,055 bytes |
| 权重 SHA-256 | `e9034d7478a8e9d1bfb57b51592e521a253287c7cdcf79258f61ea6d68584a0d` |
| 导出实现 | 89 容器 Ultralytics 8.0.111,PyTorch 2.0.0+cu118 |
| 校准 | 89 已有 COCO 图片,按文件名排序取 16 张;只用于冒烟 |
| MTK 工具 | Converter 8.16.0,NeuroPilot SDK 8.0.11,`ncc-tflite --arch=mdla5.3 --suppress-output --disallow-bridge` |
| 板端 | Genio 720 EVK,正式 v26.0,内核 6.6.137,Neuron Runtime 8.2.16 |
| 板端实现 | AArch64 C++20,OpenCV 4.9,Neuron Runtime API |
| 测试图 | 复用 YOLOv5s 公共 CC0 图 `000000000001.jpg`,1280×853 |
| 运行 ID | `20260923_cpp_fastsam_smoke_v2` |

## 产物绑定

| 文件 | SHA-256 |
| --- | --- |
| `model_fp32.onnx` | `1f5757f92c71874330f74d4dfd722ac6b93d27590ce94ec6231558d274f4143d` |
| `model_int8.tflite` | `146fba63ec8ccca1103df260033429f93d720feb340d540e7cd150ec1c2cb2cf` |
| `model_int8.dla` | `fecbd8f07d9d943d34692fae3447b578eb57a3e2234f2b36ed5ed4a10d1e41ae` |
| `runtime_config.csv` | `2771cb1a79e90be9ef69fd6773614b2b3cd3eba47f914d3e76cbf2f7e4ee504e` |
| `fastsam_board` | `d4f571e9f7de3411d236640fb02e8e7c1a31eae8e5c81aac20fe13e8afd51d0c` |
| 测试图 | `e17697994ed44c9d67280f16063eb05c58b85cd1f1761ffcb6b2fec6fafa9405` |
| 公共叠加图 | `969185af085e9590c4fcf5006f0ade1a51c6df76d4a78d69f6744b86ef9e60ae` |

导出和量化来源清单位于 89 的单模型 `models/` 目录.
板端原始证据位于
`/root/hailong.he/open_models/fastsam/runs/20260923_cpp_fastsam_smoke_v2`,
89 和本机副本位于 `examples/output/runs/20260923_cpp_fastsam_smoke_v2`.
后者被 Git 忽略,保留原始 PNG、INT8 输出、日志和 JSON 报告.

## 单图检查

| 项目 | 结果 |
| --- | ---: |
| 板端 C++ 输入与导出基线量化字节 | 完全一致 |
| C++ 单次硬件 API 调用 | 15.909384 ms |
| 模型加载 | 14.305000 ms |
| 预处理 | 27.076308 ms |
| 后处理 | 184.981693 ms |
| 单图端到端,不含输出写盘 | 243.595616 ms |
| C++ 峰值 RSS | 135252 KiB |
| 输出实例 | 100,达到本次 max_det=100 上限 |
| PyTorch 与 NPU 各取前 30 个实例 | 28 对匹配;各有 2 个未匹配 |
| 匹配实例的平均掩码 IoU | 0.956245 |
| 同一 NPU 原始输出,C++ PNG 与 NumPy 后处理 | 100 个掩码平均/最低 IoU 均为 1.000000 |

对照采用 confidence=0.4、NMS IoU=0.9、前 30 个实例,
框 IoU≥0.5 时按分数顺序做一对一匹配.未匹配实例保留计数,
不能只看匹配对的高掩码 IoU.原始张量存在 INT8 量化误差,
且 16 张校准图只为打通转换流程,不是正式精度配置.

第一次板端运行 `v1` 暴露了 1280×853 图片在 426.5 缩放高度时的取整差异:
C++ `lround` 与 Python `round` 使上下填充相差 1 个模型像素.
`v2` 改为最近偶数取整后,板端输入与导出基线逐字节相同,
100 个 C++ 掩码与同源 NPU 输出的 NumPy 复算结果一致.

![FastSAM-s 板端分割叠加图](../examples/output/public/fastsam_s_sample_1_overlay.jpg)

原始图片与 CC0 来源说明见
[YOLOv5s 公共输入](../../../../perception/object_detection/yolov5s/examples/input/public/ASSET_LICENSE.md).

## 范围

本次只执行一张图片、一次 C++ 硬件推理;单次 API 墙钟不等于稳定纯 NPU 延迟.
没有预热后多轮统计、正式标注数据集 mAP 或 G5100 验证.
后续正式评测应保持同一预处理、阈值和来源版本,
记录独立数据集、运行 ID、模型哈希及全量未匹配实例.
