# 精度报告

状态：LFW 全量十折验证已完成；应用场景精度仍待评测。

运行编号 `20260928_mobilefacenet_lfw_full_v1`，Genio 720 提取 7,701 张不同图片的特征，按 6,000 对、10 折协议验证，正确 4,270 对，准确率 `0.7116666667`。每折以其余 9 折选择余弦阈值。[完整报告](../results/full_accuracy/20260928_mobilefacenet_lfw_full_v1/summary.json)保留各折阈值及哈希。数据使用原始**非对齐**人脸并直接缩放到 112×112，不能与关键点对齐的 LFW 公布准确率直接比较。

数据源为 [LFW Hugging Face 整理版](https://huggingface.co/datasets/marcelohaps/lfw)，原始 Parquet 地址 `https://hf-mirror.com/api/datasets/marcelohaps/lfw/parquet/default/train/0.parquet`，SHA-256 `85ff8ac9530a935d2dc6f9e2933cfd72c79089b2f1c403c996b808ba5c07abcf`。验证对来自提交 `12a61458b56d0433d07269dc1d64368abf4f6b4d` 的 `pairs.csv`，SHA-256 `7f540157be42f57ab5bb1d7ef53b7b379e0331b4842cbd32f4d1b625243e54fe`。板端 `/root/hailong.he/datasets/lfw/` 共 13,233 张图、6,000 对。

2026-09-24 的[板端冒烟](smoke.md)验证了单张人脸的 PyTorch、ONNX、NPU 特征方向一致性，
以及两次相同输入的 NPU 输出一致性。此结果不构成 LFW 准确率、开放集误识率或机器人现场识别率。
机器人现场识别仍需固定检测与对齐协议、人员登记和阈值，再以不同人、不同光照和姿态的数据集评测。
