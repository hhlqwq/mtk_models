# FastSAM 公共叠加图来源

`fastsam_s_sample_1_overlay.jpg` 由本项目的 Genio 720 板端 C++ 程序基于
YOLOv5s 公共输入 `000000000001.jpg` 生成.原始图片的 CC0-1.0 授权和来源见
[YOLOv5s 样例来源](../../../../../../perception/object_detection/yolov5s/examples/input/public/ASSET_LICENSE.md).

叠加图对应运行 `20260923_cpp_fastsam_smoke_v2`,模型输出不是正式数据集精度证明.
文件 SHA-256: `969185af085e9590c4fcf5006f0ade1a51c6df76d4a78d69f6744b86ef9e60ae`.

`sample_1_segmentation.jpg`、`sample_2_segmentation.jpg` 和
`sample_3_segmentation.jpg` 为 Genio 720 板端 C++ 基于同一目录下
YOLOv5s 三张 CC0 公共输入生成的衍生可视化，输入依次为
`000000000001.jpg`、`000000000002.jpg`、`000000000003.jpg`。
对应 JSON 保存各 30 个板端实例的分数、框、掩码 RLE 和耗时。
Demo 置信度 0.4、NMS IoU 0.9、`max_det=30`；不作为正式 COCO AP 证据。

| 图片 | SHA-256 |
| --- | --- |
| `sample_1_segmentation.jpg` | `81c6c81efcdd9c4f5487395e5cb324d45b60e3896639d29654dd16a9f81f0a0b` |
| `sample_2_segmentation.jpg` | `2ae1cb48b8f853610ff6eb67741fd7bd02f427bbfe55d926a1ce730de52bd9cb` |
| `sample_3_segmentation.jpg` | `30e9127fe7c769e73ade0e237380d508c825831bf1d4fb2947d07ab2ef39bec3` |
