# RTMPose Body2d 模型卡

## 来源

- Hugging Face：<https://huggingface.co/qualcomm/RTMPose-Body2d>
- Qualcomm AI Hub Models：<https://github.com/qualcomm/ai-hub-models/tree/main/src/qai_hub_models/models/rtmpose_body2d>
- 模型版本：Qualcomm AI Hub Models v0.61.0 预导出 FP32 ONNX.

## 模型规格

- 参数量：17.9M.
- 输入分辨率：256×192.
- 输出：133 个 WholeBody 关节点.
- 来源页面许可证：Apache-2.0.

模型输入是人体框裁剪结果,不包含人体检测器.精度报告需要锁定原始数据集、人体框来源、仿射
变换和 SimCC 解码配置.
