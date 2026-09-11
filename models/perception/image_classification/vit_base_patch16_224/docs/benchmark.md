# ViT-Base 性能报告

## 正式开源模型结果

2026-09-10 在 Genio 720 EVK 使用正式 TorchVision 权重衍生 DLA 完成板端实测.

| 项目 | 数值 |
| --- | ---: |
| 连续推理次数 | 100 |
| 纯 NPU 平均延迟 | 51.7129 ms/inf |
| 纯 NPU 吞吐 | 19.31 FPS |
| 单次进程端到端耗时 | 171 ms |
| Peak RSS | 94,504 KB |

连续推理总时间为 5,171.3 ms.纯 NPU 指标不包含图片解码、resize、中心裁剪、输入
量化和 Top-5 后处理；单次进程端到端包含进程启动、模型加载、推理和输出写入.
DLA SHA-256 为
`2aec455949fa25420611645eb97617433046b6d488d87e66edf26e8246b85d67`.

> 历史结果说明：本页数据来自 Qualcomm v0.61.0 预导出 FP32 ONNX 的 MTK 衍生模型,
> 不属于待重新锁定的开源上游 ViT 正式交付结果.

```text
模型: ViT-Base Patch16 224
输入: 224x224 RGB
输出: 1000 classes
设备: MediaTek Genio 720 EVK
运行环境: NeuroPilot SDK 8.0.11 / neuronrt 8.2.16
纯 NPU 平均延迟: 53.3903 ms/inf (100 次, turbo)
纯 NPU FPS: 18.7
单次进程端到端: 171 ms (含进程启动、模型加载、推理和输出写入)
Peak RSS: 94,604 KB
```

## 测试说明

2026-09-08 在 92 开发板实测. 100 次连续推理总时间为 5339.02 ms, neuronrt
报告平均 53.3903 ms/inf 和 18.7 FPS. 该纯 NPU 指标不包含图片解码、resize、
中心裁剪、输入量化和 Top-5 后处理.

`OneShotProcessMs=171` 使用单独 neuronrt 进程测量, 包含进程启动、模型加载、
一次推理和输出文件写入, 不是稳态应用端到端延迟. 峰值 RSS 通过对 100 次推理进程的
`/proc/<pid>/status:VmHWM` 轮询获得, 为 94,604 KB.

CPU governor 为 8 核 `schedutil`; 采样时 cpu0~7 当前频率依次为
1250/1250/1250/1350/1350/1350/1750/1650 MHz. 原始日志保存在 89 的
`examples/output/{benchmark.log,memory.txt,cpu_frequency.txt,system.txt}` 和 92 的
`/root/hailong.he/vit_base_patch16_224/output/`.

历史 Qualcomm 衍生模型已满足板端运行和性能取证；这些数据不计入新的开源上游交付状态.
