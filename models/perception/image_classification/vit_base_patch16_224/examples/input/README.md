# 输入样例

`deploy_board.sh` 默认使用 89 上的 `ILSVRC2012_val_00000001.JPEG`, 并按当前
TFLite 量化参数生成 `input_int8.bin` 和 `input_metadata.json`. 原始图片不复制、
不提交, 以保持数据集不变.

执行 `bash deploy/generate_examples.sh` 会从正式 50,000 张板端评测中提取前五张
ImageNet 图片到 `generated/`.原图受 ImageNet 条款约束,默认不提交 Git.
