# 示例输出

`public/` 保存三张公开样例的 FP32 与 Genio 720 纯 NPU 检测图。两端使用同一组输入与 0.25 分数阈值，模型图在板端全部由 NeuronExecutionProvider 执行；DFL、框解码和 NMS 在 CPU 后处理。

| 样例 | FP32 | 板端纯 NPU | 检测数 |
| --- | --- | --- | ---: |
| 城市路口 | [结果](public/000000000001_fp32.jpg) | [结果](public/000000000001_npu.jpg) | 10 |
| 室内餐桌 | [结果](public/000000000002_fp32.jpg) | [结果](public/000000000002_npu.jpg) | 13 |
| 公园人物 | [结果](public/000000000003_fp32.jpg) | [结果](public/000000000003_npu.jpg) | 4 |

[逐框比较](public/comparison.json)：最大分数差 0.00535，最大坐标差 0.156 像素。
