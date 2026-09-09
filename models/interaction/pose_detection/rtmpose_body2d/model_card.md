# RTMPose Body2d 模型卡

## 当前开源上游状态

- 官方项目：<https://github.com/open-mmlab/mmpose>
- 目标变体：RTMPose-M、COCO-WholeBody 133 点、256×192 输入.
- 固定源码版本、官方配置文件、权重地址、权重 SHA-256：待官方页面核验后锁定.

锁定前不得继续使用 Qualcomm 预导出 ONNX 生成正式 MTK 产物.原始框架、ONNX 和
MTK NPU 必须使用同一个官方开源权重完成同协议精度对比.

## 历史 Qualcomm 衍生基线

- 交付形式参考：<https://huggingface.co/qualcomm/RTMPose-Body2d>
- 历史实现：<https://github.com/qualcomm/ai-hub-models/tree/v0.61.0/src/qai_hub_models/models/rtmpose_body2d>
- 历史模型：Qualcomm AI Hub Models v0.61.0 预导出 FP32 ONNX.

## 历史模型规格

- 参数量：17.9M.
- 输入分辨率：256×192.
- 原始输入张量：`image [1,3,256,192]`,NCHW RGB float32 `[0,255]`;原始图内执行
  RGB 到 BGR 及 mean/std 归一化.
- MTK 兼容输入：NCHW BGR float32 `[0,255]`;为规避 MDLA 不支持的通道 Gather,
  预处理侧完成 RGB 到 BGR,数值等价验证覆盖两个 SimCC 输出.
- 输出：`pred_x [1,133,384]`、`pred_y [1,133,512]`,SimCC split ratio 2.0.
- MMPose 仓库许可证和具体权重的第三方声明仍需随固定版本一起核验.
- 历史 Qualcomm 页面标记的许可证为 Apache-2.0,不能替代新上游许可证核验.

模型输入是人体框裁剪结果,不包含人体检测器.本项目采用 1.25 倍人体框边距并调整到
192:256 比例后执行仿射变换.精度报告需要锁定原始数据集、人体框来源、仿射变换和
SimCC 解码配置.
