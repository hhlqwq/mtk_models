# YAMNet 模型卡

- 任务: 机器人环境声音事件分类,521 类,多标签分数.
- 上游: Google TensorFlow Models / AudioSet YAMNet,版本见资源来源记录.
- 功能参考: https://aihub.qualcomm.com/models/yamnet
- 输入: `1x1x96x64` Log-Mel,默认原生 FP16,可选 INT8.
- 输出: `1x521` Sigmoid 类别分数,窗口均值用于音频级评测.
- 前处理: 16 kHz 单声道,25 ms 周期 Hann,10 ms 帧移,96 帧窗,48 帧步长.
- 部署: Genio 720 / MT8189 / MDLA 5.3,CPU 音频前处理,C++ Neuron Runtime.
- 评测: ESC-50 固定 47 类投影,第二至第五折主评测,同步报告全量指标.
- 默认部署: FP16,不使用校准数据;INT8 校准仅使用第一折.
- 范围: 环境声音分类,不是语音转文字、声源定位或设备故障诊断模型.
- 源码许可: Apache-2.0;ESC-50 整体数据许可 CC BY-NC 3.0.
- Genio 5100: 尚未部署测试.
- 实测结果: Genio 720 三后端全量完成,见 README 和 `results/summary.json`.
