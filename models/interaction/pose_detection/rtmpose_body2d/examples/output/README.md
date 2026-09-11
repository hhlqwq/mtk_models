# 输出样例

保存板端关键点可视化结果和运行摘要.

- `public/keypoints.json`:三张图片的 133 点解码结果.
- `public/sample_1_keypoints.jpg` 至 `sample_3_keypoints.jpg`:带人体框和有效关键点的
  姿态可视化图片.
- `public/backend_comparison.json`:FP32 ONNX 与 MTK NPU 的数值和关键点一致性比较.
- `board_raw/`:三张样例的双路 SimCC 原始输出,不提交 Git.

`public/` 纳入 Git,用于公开展示 Genio 720 测试结果.此外,
- `board_cpp_accuracy/<run_id>/`: 正式 WholeBody 板端逐框预测、AP/AR、耗时和输入输出哈希；
  目录默认不提交 Git，正式 v2 证据保存在 89 服务器的
  `20260909_wholebody_int8_v2` 运行目录.
