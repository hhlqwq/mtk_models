# 输出样例

保存板端关键点可视化结果和运行摘要.生成文件默认不提交 Git.
`deploy_board.sh` 会生成:

- `board_raw/`: 两张样例的双路 SimCC 原始输出、板端日志和性能/内存证据.
- `keypoints.json`: 133 点解码结果.
- `sample_*_keypoints.jpg`: 带 person 框和有效关键点的可视化.
- `backend_comparison.json`: FP32 ONNX 与 MTK NPU 的数值和关键点一致性比较.
