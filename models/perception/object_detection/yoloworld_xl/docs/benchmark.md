# 性能报告

## 官方参考

MediaTek IoT AI Hub Model Zoo 记录的输入为 `3×640×640`、Float32：

| 平台 | 执行后端 | 官方纯模型延迟 |
| --- | --- | ---: |
| Genio 720 | Neuron EP | 403.15 ms |
| Genio 720 | CPU EP | 11214.33 ms |

官方数据由 `onnxruntime_perf_test` 测量,不能作为本项目实测结果.

## 本项目实测

2026-09-16 在 Genio 720 上使用官方 v26.0 rootfs 提取的 Neuron `8.2.16` adapter/runtime
隔离运行.最终采用完整 opset 13 模型和 `NEURON_FLAG_MIN_GROUP_SIZE=100`,保留 CPU
fallback 以保证检测正确性.本轮为三张公开图片的小样本验证：Neuron 每张图预热 1 次后
重复 3 次,共 9 次；CPU 每张图运行 1 次,共 3 次.

| 项目 | CPU EP | 混合 Neuron EP |
| --- | ---: | ---: |
| mean | `8334.186 ms` | `3720.758 ms` |
| min | `8319.619 ms` | `3699.040 ms` |
| max | `8350.524 ms` | `3743.153 ms` |
| P50 | `8332.416 ms` | `3719.365 ms` |
| P90 | `8346.902 ms` | `3735.285 ms` |
| P95 | `8348.713 ms` | `3738.382 ms` |
| 峰值 RSS | `830908 KiB` | `1126744 KiB` |

混合路径相对本轮 CPU mean 加速 `2.2399×`.30 次正式推理 profiling 记录 330 个 Neuron
节点事件和 6336 个 CPU 节点事件,因此该结果不能表述为全模型 NPU.完整结构化证据见
`board_validation_20260916.json`.

当 `MIN_GROUP_SIZE=0` 时,完整 opset 13 模型可达到 `401.500 ms` mean,与官网 `403.15 ms`
接近,但三张图片都触及 300 个检测上限且高分结果饱和到 `1.0`,未通过正确性门禁,不得作为
有效性能结果发布.正式性能轮次仍需使用正确性通过的配置、固定性能模式、至少 3 次预热和
每图 10 次以上重复.

## 运行时诊断

板端原 `/usr/lib` 中的同名 Neuron `8.2.16` 库会让 YOLO-World 和系统自带
`squeezenet_quant.onnx` 同样在建图阶段失败.换用官方 v26.0 rootfs 内的隔离库后,
SqueezeNet 单次 NPU 推理为 `12.62 ms`,YOLO-World 也成功执行.因此此前
`unregistered target: NEON` 是板端运行库混装/构建差异,不是 YOLO-World 专属算子问题.

日志仍会打印缺少 `libnir_neon_driver.so` 及若干 `unregistered target: NEON` 警告,但官方
adapter 能将可执行子图分配到 MDLA 并完成会话；不能仅凭这些警告判定 NPU 失败,应以进程
退出码、非空结果和 ORT profiling 的 Neuron 节点为准.
