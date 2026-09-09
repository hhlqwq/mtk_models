# RTMPose Body2d 模型卡

## 来源

- Hugging Face：<https://huggingface.co/qualcomm/RTMPose-Body2d>
- Qualcomm AI Hub Models：<https://github.com/qualcomm/ai-hub-models/tree/main/src/qai_hub_models/models/rtmpose_body2d>
- 模型版本：Qualcomm AI Hub Models v0.61.0 预导出 FP32 ONNX.

## 模型规格

- 参数量：17.9M.
- 输入分辨率：256×192.
- 输入张量：`image [1,3,256,192]`,NCHW RGB float32 `[0,1]`;模型图内执行
  RGB 到 BGR 及 mean/std 归一化.
- 输出：`pred_x [1,133,384]`、`pred_y [1,133,512]`,SimCC split ratio 2.0.
- 来源页面许可证：Apache-2.0.

模型输入是人体框裁剪结果,不包含人体检测器.本项目采用 1.25 倍人体框边距并调整到
192:256 比例后执行仿射变换.精度报告需要锁定原始数据集、人体框来源、仿射变换和
SimCC 解码配置.
