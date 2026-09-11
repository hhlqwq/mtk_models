# 输出样例

保存板端关键点可视化结果和运行摘要.生成文件默认不提交 Git.
`deploy_board.sh` 会生成:

- `board_raw/`: 两张样例的双路 SimCC 原始输出、板端日志和性能/内存证据.
- `generated/keypoints.json`: 五张图片的 133 点解码结果.
- `generated/sample_1_keypoints.jpg` 至 `sample_5_keypoints.jpg`: 带 person 框和
  有效关键点的五张姿态可视化图片.
- `generated/backend_comparison.json`: FP32 ONNX 与 MTK NPU 的数值和关键点一致性比较.
- `board_cpp_accuracy/<run_id>/`: 正式 WholeBody 板端逐框预测、AP/AR、耗时和输入输出哈希；
  目录默认不提交 Git，正式 v2 证据保存在 89 服务器的
  `20260909_wholebody_int8_v2` 运行目录.
