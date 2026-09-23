# YOLO-World XL 板端 Demo

`run_board.py` 使用板端预装的 ONNX Runtime 1.20.2.`neuron` 模式设置官方 benchmark
同款的 `NEURON_FLAG_USE_FP16=1`、`--opt=3`、
`--num-mdla=1`、`--reshape-to-4d` 和 `--interval-coloring-converage=1.0`,并把 CPU EP
保留为不支持算子的回退后端.本模型默认将 `NEURON_FLAG_MIN_GROUP_SIZE` 提高到 `100`,
以避开检测头全量下沉 NPU 时的数值漂移.

模型输出是三个尺度上的 COCO 80 类 logits 与四方向距离,Demo 在 CPU 上完成 Sigmoid、
stride 8/16/32 距离框解码、逐类别 NMS 和原图坐标恢复.

建议通过模型根目录的 `deploy/run_board.sh` 执行,以保存系统、模型和输出哈希证据.
`results.json` 会为每张图记录六个输出张量的 shape、min、max、mean 和有限值比例,用于
定位 CPU 与 Neuron EP 的分类头或框头数值漂移,但不会保存体积较大的完整输出张量.
`--neuron-min-group-size` 默认值为本项目实测通过的 `100`.设为官方 benchmark 的 `0`
虽然接近官网纯性能,但三张样例均会触及 300 个检测上限,不得作为正确结果发布.
如果板端系统运行库与官方 v26.0 不一致,先运行 `deploy/stage_neuron_runtime.sh`,再设置
`MTK_NEURON_RUNTIME_DIR=/root/hailong.he/open_models/yoloworld_xl/runtime_v26`.隔离目录不会覆盖
`/usr/lib`.
