# YOLOv8n / Genio 720

COCO 80 类目标检测,固定输入 `1x3x640x640` RGB,INT8 PTQ,目标为
MT8189 / MDLA 5.3.当前状态: 环境建设中,代码已实现,转换及板端结果尚未验证.

## 官方来源

- 实现: https://github.com/ultralytics/ultralytics/tree/v8.0.111
- 安装版本: `ultralytics==8.0.111`,复用现有 Docker 包,不自动升级.
- 权重: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
- 权重 SHA256: `31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95`.
- 数据集: https://cocodataset.org/#download ,使用现有 COCO val2017.
- Qualcomm 仅作为交付结构参考,本模型不使用其预导出产物.

## 两步运行

先在 Windows 修改并提交代码,89 服务器拉取.将上述官方权重离线放入
`models/yolov8n.pt`,脚本会验证哈希,不会在线下载模型和依赖.

编译主机 `ubuntu89`,工作目录 `/data/users/hailong.he/github/mtk_models`:

```bash
bash models/perception/object_detection/yolov8n/deploy/run.sh
```

默认使用 Docker `hhl_g720_8011`,NeuroPilot SDK 8.0.11,MTK Converter 8.16.0,
100 张固定排序的 COCO 图片校准.脚本完成 ONNX 导出、INT8 TFLite 转换、
`ncc-tflite --arch=mdla5.3 --suppress-output --disallow-bridge` 编译,
PyTorch 与 ONNX 的 5000 张同协议精度评测、C++ 交叉编译和上传.
模型产物保存在模型的 `models/`,临时文件保存在
`/tmp/hailongcodex/当天日期/yolov8n`.

开发板 `root@192.168.0.92`:

```bash
bash /root/hailong.he/open_models/yolov8n/run.sh
```

板端使用已有 `/root/hailong.he/datasets/coco/val2017`,执行 20 次预热和
5000 张真实 NPU 推理,计算 mAP、平均 NPU 延迟、端到端延迟及进程峰值 RSS,
保存实际检测效果.结果位于部署目录 `results/运行ID/`.
不同运行使用独立目录,保留失败日志、预测、耗时和模型哈希,不自动删除.

## 适配与评测协议

三个尺度分别导出 64 通道框分布和 80 通道分类 logits,共六个输出.
DFL、Sigmoid、类别内 NMS 和原图坐标还原在 CPU 执行.
量化参数及输出索引从本次 TFLite 解析,通过 `runtime_config.csv` 部署;
Runtime 检查缓冲区尺寸,支持 NCHW 行宽按 16 对齐.
每次推理重新登记输入和全部输出,避免复用陈旧缓冲区.

PyTorch、ONNX 和板端使用相同 letterbox、单最佳类别、`conf=0.001`,
`IoU=0.6`,`max_det=300`.该协议用于本项目三后端对比,
不直接等同于 Ultralytics 官方公开指标的复现协议.
公开示例取自已有 COCO 数据,许可证和原始地址见
`examples/input/samples.json`,展示阈值为 0.25.

## 当前结果

尚未生成本次运行结果,不预填精度或性能数字.Genio 5100 尚未执行.
