# YOLOv5s 三图示例

`input/` 中三张图片由项目维护者于 2026-09-11 使用 OpenAI 图像生成工具生成，不取自 COCO、ImageNet 或其他第三方数据集。项目维护者按 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) 发布，可复制、修改和分发。它们只用于展示板端推理，不用于正式精度评测。

| 文件 | 场景提示词摘要 |
| --- | --- |
| `000000000001.jpg` | 城市路口、行人、自行车、汽车和交通灯。 |
| `000000000002.jpg` | 餐厅、桌椅、盆栽、杯子和瓶子。 |
| `000000000003.jpg` | 公园内人物遛狗，附近有长椅和自行车。 |

仓库图片统一缩放到长边 1280 像素并保存为质量 88 的 JPEG。`output/` 保存三图展示图片；`timing_summary_current_run.json` 是历史 5000 张全量运行摘要，三图冒烟会另写 `smoke_timing_summary.json`。正式 COCO 精度另见模型的 `docs/accuracy.md`。
