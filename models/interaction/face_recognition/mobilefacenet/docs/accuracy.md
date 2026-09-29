# 精度报告

状态：输入协议纠正后，LFW 全量十折精度重测中；应用场景精度仍待评测。

锁定上游提交的 [`lfw_eval.py`](https://github.com/foamliu/MobileFaceNet/blob/a687c71bea830e70d05fb3b38ddc7c68e1687e94/lfw_eval.py) 先做人脸五点对齐，将 BGR 转为 RGB，再使用 [`data_gen.py`](https://github.com/foamliu/MobileFaceNet/blob/a687c71bea830e70d05fb3b38ddc7c68e1687e94/data_gen.py) 的 ImageNet 均值及标准差归一化。旧运行 `20260928_mobilefacenet_lfw_full_v1` 使用 BGR、`(x - 127.5) / 128`，且对原始非对齐人脸直接缩放。因此旧 FP32 ONNX `0.7095`、板端 `0.7116666667` 只用于定位输入错误，不作为模型正式精度或与公开数值比较。原始[板端报告](../results/full_accuracy/20260928_mobilefacenet_lfw_full_v1/summary.json)和[参考报告](../results/reference_accuracy/lfw_fp32_v1/summary.json)保留供复核。

数据源为 [LFW Hugging Face 整理版](https://huggingface.co/datasets/marcelohaps/lfw)，原始 Parquet 地址 `https://hf-mirror.com/api/datasets/marcelohaps/lfw/parquet/default/train/0.parquet`，SHA-256 `85ff8ac9530a935d2dc6f9e2933cfd72c79089b2f1c403c996b808ba5c07abcf`。验证对来自提交 `12a61458b56d0433d07269dc1d64368abf4f6b4d` 的 `pairs.csv`，SHA-256 `7f540157be42f57ab5bb1d7ef53b7b379e0331b4842cbd32f4d1b625243e54fe`。板端 `/root/hailong.he/datasets/lfw/` 共 13,233 张图、6,000 对。

2026-09-24 的[板端冒烟](smoke.md)验证了单张人脸的 PyTorch、ONNX、NPU 特征方向一致性，
以及两次相同输入的 NPU 输出一致性。此结果不构成 LFW 准确率、开放集误识率或机器人现场识别率。
机器人现场识别仍需固定检测与对齐协议、人员登记和阈值，再以不同人、不同光照和姿态的数据集评测。
