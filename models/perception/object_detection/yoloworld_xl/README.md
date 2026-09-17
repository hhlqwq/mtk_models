# YOLO-World XL

## 模型信息

```text
模型: YOLO-World XL
任务: 固化 COCO 80 类文本嵌入的目标检测
输入: images [1,3,640,640],NCHW RGB float32 [0,1]
输出: 三尺度类别 logits 与四方向框距离
设备: MediaTek Genio 720 EVK
部署格式: ONNX Runtime + Neuron Execution Provider
当前状态: 板端 Neuron EP 小样本已验证,正式 COCO mAP 待执行
```

本交付实现用户指定的 MediaTek IoT AI Hub 官方 Model Zoo ONNX,而不是把其他平台的
预转换模型作为移植源.算法上游为 AILab-CVC/YOLO-World,固定提交
`4f70adbaacf5685bd9ec5bea85f1f91057f6fc0b`；MediaTek 官方 ONNX 是本次明确指定的
板端部署资产.官方文件没有携带自定义 metadata,因此仓库不会声称已经证明该二进制由
哪个上游 checkpoint 导出.

## 官方模型下载

下载地址：

<https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/iot-ai-hub/model-zoo/onnx/robotic/yoloworld_xl.onnx>

本机放置位置：

```text
D:\code\github\mtk_models\models\perception\object_detection\yoloworld_xl\models\model_fp32.onnx
```

| 字段 | 实测值 |
| --- | --- |
| 文件大小 | `419,032,016 bytes` |
| SHA-256 | `6d5b231425200f0426b73967c33a69ce5af83e993426d4fc64dc1709571d6174` |
| Last-Modified | `2026-01-17T08:37:58Z` |
| S3 ETag | `532696e4dcbb6bbe8589e1f8bd4e7d23-25`,分段上传标识,不是 MD5 |

模型和生成产物由 `.gitignore` 排除,不进入普通 Git 历史.89 和 92 不从公网下载该文件；
模型先下载到 Windows 工作区,再离线传到服务器和开发板.

## 为什么需要 opset 13 兼容模型

官方 ONNX 是 opset 11,包含三个 `axis=3` 的 Softmax.当前板端 ORT 1.20.2 的 Neuron EP
明确拒绝版本低于 13 的 Softmax,原始模型首次建图随后因 `unregistered target: NEON`
退出,不能算作 NPU 推理成功.

`deploy/prepare_onnx.py` 使用 ONNX 官方 version converter 将完整图转换到 opset 13,
避免只修改 opset 标头造成 Squeeze/Unsqueeze 等算子语义不一致.转换后必须通过 ONNX
checker,并由 `verify_onnx_equivalence.py` 在 CPU EP 上使用同一个确定性随机输入验证六个
输出逐元素完全一致,才能部署.

## 服务器准备流程

89 宿主机和 Docker 内仓库路径保持一致：

```bash
cd /data/users/hailong.he/github/mtk_models
docker exec -it hhl_g720_8011 bash
cd /data/users/hailong.he/github/mtk_models/models/perception/object_detection/yoloworld_xl
bash deploy/build.sh
```

这里的 `build.sh` 仅生成并验证在线推理使用的 opset 13 ONNX,不运行 MTK Converter、
NCC,也不生成 DLA.

## 板端运行

### 当前板端运行库差异

当前板端 `/usr/lib` 的 Neuron `8.2.16` 与官方 Rity v26.0 镜像中的同版本二进制并不
相同.系统 adapter Build ID 为 `5b1e4ff9083d05ce2da479aa0dfd21b51ce1d7b1`,会让
YOLO-World 以及系统自带 `squeezenet_quant.onnx` 都以 `unregistered target: NEON`
退出；因此该错误不能归因于 YOLO-World 图.

官方 v26.0 rootfs 内的 adapter Build ID 为
`4799fa69bc82519b5ed3e0392ae566cc28ed9f42`.本项目不覆盖系统库,而是把官方镜像内的
adapter/runtime 提取到模型专用目录.在 89 上执行：

```bash
bash deploy/stage_neuron_runtime.sh \
  /tmp/hailongcodex/20260916/rootfs/9.rootfs.img
```

脚本只接受已经从 MediaTek 官方 v26.0 镜像解出的 ext4 rootfs,并校验以下哈希：

| 文件 | SHA-256 |
| --- | --- |
| `9.rootfs.img` | `ce857239c548dd66c65cfef49f809582c107405c597c6773f64115fa03650c9d` |
| `libneuronusdk_adapter.mtk.so.8.2.16` | `225f70c7fc5fb6fefa1ef8b662df7036ed36431b57252379ce0dd4b9e2210bcb` |
| `libneuronusdk_runtime.mtk.so.8.2.16` | `38d74829a50f802eba0539d9c37273e7743e60d692b2af3fb6054805d08a85de` |

这些专有运行库不进入 Git.正式刷入官方 v26.0 镜像后可不设置隔离目录,但仍需先用系统
自带 SqueezeNet 对照确认 Neuron EP 可用.

### 全量 NPU 检测错误根因

`NEURON_FLAG_MIN_GROUP_SIZE=0` 可以得到约 `401.500 ms` 的官网同档延迟,但其检测结果
错误不是普通 FP16 舍入误差.2026-09-17 使用固定图片和固定中间张量进行算子级拆分后,
已经把首个确定性错误定位到三个分类分支 `cls_contrasts.{0,1,2}/MatMul` 使用的
`BatchMatMulLayer`.该算子计算视觉特征与固化文本特征的相似度；stride 8 分支的输入为
`[1,6400,512] × [1,512,80]`,随后结果还会乘以 `exp(logit_scale)` 并加 bias,因此
MatMul 偏差会被放大为错误的类别 logits 和饱和置信度.

官方 [G720 Supported Operations](https://neuropilot.mediatek.com/sphinx/g720/html/l1_supported_operations/l2_supported_operations/l3_supported_ops/supported_operations.html)
把 `BATCH_MATMUL` 列为 MDLA 3.5、MDLA 5.0/5.1/5.3/5.5 和 MVPU 2.5 支持的硬件
算子,但“列表支持”不等于当前 BSP、驱动和编译器组合已经通过该形状的数值一致性验证.
当前板端的实测证据如下：

| 检查项 | 结果 |
| --- | --- |
| 单算子 CPU 输出 | range `[-24.918762,8.697776]`,mean `-4.840788` |
| 单算子 Neuron 输出 | range `[-11.875,5.464844]`,mean `-1.107744` |
| CPU/Neuron 差异 | max abs `28.957825`,mean abs `4.023510`,P99 abs `9.533292` |
| 标准 FP16 模拟误差 | 128 个位置 max abs `0.003028`,mean abs `0.000625` |
| 改写为二维 MatMul | 错误完全保留,max abs `28.957829` |
| 去掉 `--reshape-to-4d` | 错误完全保留 |
| 把 `--opt=3` 改为 `--opt=0` | 错误完全保留 |

ORT profiling 确认该微型 MatMul 由 `NeuronExecutionProvider` 执行,不是 CPU fallback.
关闭 `NEURON_FLAG_USE_FP16` 后,编译器对同一个 `BatchMatMulLayer` 明确报告 GPU/EDMA
不支持、MDLA 不支持 Float32 输入和输出,随后编译失败；这证明当前可执行路径是 MDLA
FP16.因此目前可以确认的直接根因是：**Neuron 8.2.16 在当前 Genio 720 软件栈上为
MDLA 编译或执行该大矩阵 FP16 BatchMatMul 时产生了错误数值**.问题不在预处理、NMS、
DFL、opset 11 到 13 转换,也不由 `--reshape-to-4d` 或 `--opt=3` 单独触发.

Genio 720 芯片包含 MVPU 资源,但当前环境并没有把 MVPU 暴露成可用的 Neuron device：

- Neuron 只枚举到 `mtk-gpu` 和 `mtk-mdla`,没有 `mtk-mvpu`.
- `/sys/bus/platform/drivers/mtk_mdla/` 已绑定 `soc:mdla`,而 `mtk_mvpu/` 没有绑定设备.
- 固件创建了 MVPU TX/RX RPMsg endpoint,但 `/sys/bus/rpmsg/drivers/` 没有对应 MVPU
  driver；这只能证明固件端点存在,不能证明用户态可以调度 MVPU.
- 显式使用 `--num-mdla=0 --num-mvpu=1` 时运行时异常退出,不能建立 MVPU 执行路径.

所以“板上有 MVPU”和“当前 Neuron EP 可以使用 MVPU”必须分开表述.目前没有证据证明
MVPU 2.5 执行同一 MatMul 一定正确,也没有证据证明官网 `403.15 ms` 使用了 MVPU；需要
恢复完整匹配且能够枚举 `mtk-mvpu` 的 BSP/驱动/固件/Neuron 组合后,重新运行这个单算子
门禁才能判断.在此之前,全量 NPU 结果不得用于发布；当前正确性方案仍是
`NEURON_FLAG_MIN_GROUP_SIZE=100`,让包含分类 MatMul 的 neck/head 留在 CPU.

退出容器,在 89 宿主机执行：

```bash
cd /data/users/hailong.he/github/mtk_models/models/perception/object_detection/yoloworld_xl
RUN_ID=20260916_v26isolated_v2 \
MTK_NEURON_RUNTIME_DIR=/root/hailong.he/yoloworld_xl/runtime_v26 \
bash deploy/run_board.sh
```

脚本会：

1. 部署兼容 ONNX、Python Demo 和三张公开图片.
2. 运行 CPU EP 三图基线.
3. 使用 `NEURON_FLAG_USE_FP16=1` 和正确性优先的 `MIN_GROUP_SIZE=100` 运行混合 Neuron EP.
4. 自动比较 CPU/Neuron 的检测数量、类别、分数和框坐标,超阈值即失败.
5. 保存 ORT profiling,区分 Neuron 节点和 CPU fallback 节点.
6. 回传检测框图片、JSON、延迟、峰值 RSS、环境及哈希证据.

板端目录固定为 `/root/hailong.he/yoloworld_xl/{model,demo,eval}`.每次运行必须使用新的
`RUN_ID`,不得覆盖旧证据.

## 已确认的模型接口

| 顺序 | 输出形状 | 含义 |
| ---: | --- | --- |
| 0 | `[1,80,80,80]` | stride 8 的 80 类 logits |
| 1 | `[1,4,80,80]` | stride 8 的 l/t/r/b 距离 |
| 2 | `[1,80,40,40]` | stride 16 的 80 类 logits |
| 3 | `[1,4,40,40]` | stride 16 的 l/t/r/b 距离 |
| 4 | `[1,80,20,20]` | stride 32 的 80 类 logits |
| 5 | `[1,4,20,20]` | stride 32 的 l/t/r/b 距离 |

Demo 使用与上游 ONNX 示例一致的居中黑边方形填充、RGB、除以 255 预处理；随后执行
Sigmoid、距离框解码、逐类别 NMS 和原图坐标恢复.

## 验收边界

MediaTek 官网的 Genio 720 Neuron EP `403.15 ms` 和 CPU EP `11214.33 ms` 是官方
`onnxruntime_perf_test` 纯模型参考值,不是本项目实测,也不包含前后处理.本项目只有在
隔离官方运行库已完成真实图片 NPU 小样本推理；当前实测与边界见 `docs/benchmark.md` 和
`docs/accuracy.md`,正式运行摘要见 `docs/board_validation_20260916.json`.当前标记为
"板端已验证",正式 COCO mAP 完成前不会标记为"完整交付".
