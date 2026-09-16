# YOLO-World XL

## 模型信息

```text
模型: YOLO-World XL
任务: 固化 COCO 80 类文本嵌入的目标检测
输入: images [1,3,640,640],NCHW RGB float32 [0,1]
输出: 三尺度类别 logits 与四方向框距离
设备: MediaTek Genio 720 EVK
部署格式: ONNX Runtime + Neuron Execution Provider
当前状态: 环境建设中
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

退出容器,在 89 宿主机执行：

```bash
cd /data/users/hailong.he/github/mtk_models/models/perception/object_detection/yoloworld_xl
RUN_ID=20260915_public_v1 bash deploy/run_board.sh
```

脚本会：

1. 部署兼容 ONNX、Python Demo 和三张公开图片.
2. 运行 CPU EP 三图基线.
3. 使用 `NEURON_FLAG_USE_FP16=1` 运行 Neuron EP.
4. 保存 ORT profiling,区分 Neuron 节点和 CPU fallback 节点.
5. 回传检测框图片、JSON、延迟、峰值 RSS、环境及哈希证据.

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
profiling 出现 Neuron EP 节点、真实图片输出合理并形成可复现耗时证据后,才更新为
"板端已验证".正式 COCO mAP 完成前不会标记为"完整交付".
