# RTMPose Body2d 模型卡

## 正式开源上游

- 官方项目：<https://github.com/open-mmlab/mmpose>
- 目标变体：RTMPose-M、COCO-WholeBody 133 点、256×192 输入.
- 固定版本：`v1.3.2`.
- 官方配置：`configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py`.
- 官方权重：<https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth>
- 文件大小：72,010,049 bytes.
- SHA-256：`3da02694cd6479d3b333ff42ebd0723f96bfa06adac1db1e2e815ed2e9e1b02d`.
- MD5：`f9e98931f38c2ef2acff811f479197a2`,与官方服务器 `ETag` 一致.
- 许可证：MMPose Apache-2.0；正式发布时仍需保留权重附带的第三方声明.

`deploy/export_onnx.py` 从本地官方 `.pth` 自行导出双 SimCC 输出 ONNX.原始框架、
ONNX 和 MTK NPU 必须使用同一个官方开源权重完成同协议精度对比.

## 历史 Qualcomm 衍生基线

- 交付形式参考：<https://huggingface.co/qualcomm/RTMPose-Body2d>
- 历史实现：<https://github.com/qualcomm/ai-hub-models/tree/v0.61.0/src/qai_hub_models/models/rtmpose_body2d>
- 历史模型：Qualcomm AI Hub Models v0.61.0 预导出 FP32 ONNX.

## 正式模型规格

- 参数量：17.9M.
- 输入分辨率：256×192.
- 正式输入张量：`image [1,3,256,192]`,NCHW RGB float32 `[0,255]`.
- 导出图内使用 MMPose 官方 ImageNet mean/std 完成归一化,不包含 RGB/BGR 通道重排.
- 输出：`pred_x [1,133,384]`、`pred_y [1,133,512]`,SimCC split ratio 2.0.
- 官方基准：COCO-WholeBody Whole AP 0.582、Whole AR 0.674.

模型输入是人体框裁剪结果,不包含人体检测器.本项目采用 1.25 倍人体框边距并调整到
192:256 比例后执行仿射变换.精度报告需要锁定原始数据集、人体框来源、仿射变换和
SimCC 解码配置.

## 正式 Genio 720 交付结果

| 产物 | 大小 (bytes) | SHA-256 |
| --- | ---: | --- |
| `model_fp32.onnx` | 71,832,505 | `0b4a8d276efc93b95da46b3f05c30a253ce19b83aa51e4b9c46d6f957d448684` |
| `model_mtk_compatible.onnx` | 71,967,810 | `ff148b50ae3dd610bc5f4a5c22d547ad621e031cb66090c6ef2c29ed882873d3` |
| `model_int8.tflite` | 18,741,240 | `28e96c3e0c18d127d9216d55a43fcf3300738553b52df0137910684e45e41c10` |
| `model_int8.dla` | 19,144,561 | `a5851c2e9602a17f703dbcaa2a80fd01e13bd99c11d7b2437a1e286374ed1461` |

COCO-WholeBody 正式板端结果为 WholeBody AP/AR 0.5324/0.6413,平均 NPU 延迟
3.8527 ms/框.完整结果见 `docs/board_validation_20260911.json`.
