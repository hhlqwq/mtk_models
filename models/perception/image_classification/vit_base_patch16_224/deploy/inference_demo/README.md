# ViT 板端 Demo

当前 Demo 面向从 PyTorch Vision v0.15.1 官方权重自行导出的模型.I/O 规范为：

- 输入 `image`：1×3×224×224 NCHW RGB float32,取值范围 `[0,1]`（即 rgb/255）.
- TorchVision ImageNet mean/std 归一化已内置在自行导出的 ONNX 图中,外部禁止重复归一化.
- 几何预处理：短边 resize 256（INTER_CUBIC）→ 中心裁剪 224 → BGR→RGB → /255.
- 输出 `logits`：1×1000 ImageNet logits；类别索引必须使用正式 ImageNet 映射.

`prepare_input.py` 按 INT8 TFLite 输入量化参数（scale/zero_point）生成板端 bin 和元数据；
`postprocess_top5.py` 反量化 neuronrt 原生 INT8 输出并给出 top-5,可用 `--labels` 附带类名.

通常直接在容器内执行 `bash deploy/deploy_board.sh`. 脚本默认使用 ImageNet val
第一张图片, 也可通过 `VIT_DEMO_IMAGE=/path/to/image.JPEG` 指定图片; 它会按当前
TFLite 重新生成输入, 在 92 上完成冒烟、延迟和峰值 RSS 采样, 回传输出后生成
`examples/output/top5.json`.

新 ONNX、TFLite 和 DLA 尚未生成或板端验证.历史 Qualcomm 衍生结果不能用于声明
当前开源上游模型已完成交付.

板端连接使用 `BatchMode=yes` 并在首次连接时接受 92 的主机密钥; 后续连接由
容器内 `known_hosts` 校验. 若 92 重刷系统导致主机密钥变化, 脚本会停止并要求人工核对,
不会静默忽略密钥冲突.
