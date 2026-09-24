# Depth Anything V2 Small / Genio 720 板端冒烟记录

## 结论

2026-09-24，运行编号 `20260924T092605Z`。官方 Small 权重完成固定 518×518 ONNX 导出、MTK INT8 转换、MDLA 5.3 无桥接编译，并在 92 板端执行三次 `neuronrt -m hw` 硬件推理。两张不同图片产生不同深度图，重复第一张图片的原始输出逐字节一致。状态为 `board_verified`；未完成正式深度精度和稳定性能测试。

## 来源与环境

| 项目 | 本次实际值 |
| --- | --- |
| 官方源码 | `DepthAnything/Depth-Anything-V2`，提交 `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` |
| 官方 Small 权重 | Hugging Face 版本 `03876f8651c73a60fe4c2c48294e09fcb6838fcf` 的 `depth_anything_v2_vits.pth` |
| 实际下载 | 89 无法连接 Hugging Face 官方域名，改用同版本 hf-mirror 地址；99,218,434 bytes，SHA-256 与镜像响应 `x-linked-etag` 一致 |
| 许可 | 官方 README 与模型卡标注 Small 为 Apache-2.0 |
| 转换 | Ubuntu 89，`hhl_g720_8011`，PyTorch 2.0.0+cu118，MTK Converter 8.16.0；COCO val2017 按文件名排序取 16 张图片校准 |
| 编译 | `ncc-tflite --arch=mdla5.3 --suppress-input --suppress-output --disallow-bridge` |
| 板端 | Genio 720 EVK，aarch64，Linux 6.6.137，Neuron Runtime 8.2.16 |
| 输入图 | 复用 YOLOv5s 公共 CC0 图 `000000000001.jpg` 和 `000000000002.jpg` |

官方来源和具体下载地址另见 [source_url.txt](../original/source_url.txt)。

## 模型与输入输出

| 文件 | SHA-256 |
| --- | --- |
| `depth_anything_v2_vits.pth` | `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378` |
| `model_fp32.onnx` | `cddc4399ad9155dc4380cb533aa7032520d91a19cff72e2bbd121f9318e1829e` |
| `model_int8.tflite` | `fb36f40f837c587410eba1f2cab20ab6cd9a41bffae628d3ab29bec1ee566054` |
| `model_int8.dla` | `8319d41c803e37e50d249fc3a729b007ffe27e214e8af1d4997e4849692e3dce` |

输入为 RGB、ImageNet 均值和标准差归一化后的 NCHW FP32，转换成 INT8 后传给板端。固定 518×518 形状保留了官方默认边长，但这里把不同宽高比的图片缩放成正方形。原生 MDLA 输入每行补齐到 528 字节，因此实际 `.bin` 为 `3×518×528 = 820,512` 字节；模型输出是 518×518 INT8，相应 `.bin` 为 268,324 字节。输出数值是相对深度，不能解释为米。

为避免 MTK Converter 不支持的位置编码 bicubic Resize，导出时预先计算固定 518×518 的位置编码；同时用官方 QKV 和投影权重改写为四维注意力。固定随机输入上，两处改写前后 PyTorch 输出最大绝对差为 0。ONNX 使用 opset 17；该数值检查只覆盖固定输入形状。

## 板端核验

`ncc-tflite --show-exec-plan` 显示图只有一个执行步骤，目标为 `MDLA_5_3`。板端三次均通过 `neuronrt -m hw` 生成非空输出，报告如下：

| 检查 | 图片 1 | 图片 2 |
| --- | ---: | ---: |
| 输入 SHA-256 | `4a298dcc12fdbc5c43ef794c49ef2f1efdc1fc232033b199b57f0862200dbff3` | `fa2afe9d6fa666802377fa673798536b58eac02b01c23b4264731d7663a53a66` |
| 输出 SHA-256 | `9b58056c5b6d7a59aaae89ddf7a23e319752310dfeaabc3c8a1649aa54eb8095` | `b207ef1c5f911f00dbfb984fd46011f614d4f38e572be6bc26a13da7fe26c84d` |
| PyTorch 与板端逐像素 Pearson 相关系数 | 0.995495 | 0.990525 |
| 板端反量化深度范围 | 0–5.554508 | 0–3.848221 |

第三次输入和输出分别与第一次的 SHA-256 完全一致。Pearson 相关系数只反映空间变化方向；图片 1 的 PyTorch 最大值为 8.493637，板端为 5.554508，说明绝对数值存在明显量化偏差，不能据此宣称正式精度达标。
冒烟核验脚本要求每张图片的相关系数至少为 0.9，并要求输出有限、非恒定、不同图片不相同。

![板端相对深度预览](../examples/output/public/depth_anything_v2_small_sample_1.png)

预览对单图深度值做了 2%–98% 分位数拉伸，仅用于检查输出结构；原图来源和 CC0 许可见 [公共输入说明](../../../../perception/object_detection/yolov5s/examples/input/public/ASSET_LICENSE.md)。

## 证据位置与边界

- 92 原始目录：`/root/hailong.he/open_models/depth_anything_v2_small/smoke/20260924T092605Z/`。
- 89 副本：`/data/users/hailong.he/github/mtk_models/models/navigation/single_camera_depth/depth_anything_v2_small/examples/output/runs/20260924T092605Z/`，含输入、原始输出、日志、哈希及 `output/report.json`。
- 89 的 `models/exec_plan_518.log` 记录单个 MDLA 执行步骤；`models/build_518.log` 和 `models/build_518_native.log` 记录编译尝试。

首次 518×518 板端执行因原生输入未补齐而报告需要 820,512 字节。修正行步长后，三次真实硬件推理通过。更早的 252×252 探针导出曾因官方注意力形成五维张量而无法无桥接编译；四维等价改写后解决该限制。

本次只验证两张图片、三次调用。未评估 DA-2K 等正式数据集精度、量化配置、预热后的延迟、内存峰值、视频时序稳定性和 Genio 5100。相对深度不应用作实际米制距离或安全避障阈值。
