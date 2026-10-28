# MiniCPM5-2B

面向 MediaTek Genio 720 / MT8189 的文本生成模型,使用官方 [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B),许可证 Apache-2.0.来源见 [source_url.txt](models/source_url.txt).

| 项目 | 当前配置 |
|---|---|
| 结构 | 42 层,hidden size=2048,Query/KV heads=16/2 |
| 量化 | 全 W4A16,包含 LM Head;外置 embedding 为 INT16 |
| 配方 | 16 批次 Hessian 重构 + Hadamard 旋转 |
| 上下文 / Prefill | 1024 / 128 Token,Decode 每步 1 Token |
| 后端 | Python 直接调用 MTK Neuron Runtime,MDLA 5.3 |
| 生成 | Greedy,关闭 thinking,停止 Token 为 1/130073 |

当前默认使用已验证并固化的量化图、双 DLA 与配套资源.固定 16 题 MMLU 子集答对 **8/16**,同题浮点参考为 **13/16**;该子集不能代表整体 MMLU,也不证明已经达到量化最优.当前 Python 实现的正式范围精度、性能和 RSS 尚未完成评价.过程、对照与历史记录见 [ADAPTATION.md](ADAPTATION.md);[summary.json](results/summary.json) 明确保留迁移前历史报告.

## 使用

参数和离线资源位置由 [config.yaml](config.yaml) 及本地平台配置管理,运行时不自动下载.在已配置的编译容器中,进入本模型目录执行:

```bash
bash deploy/compile.sh
```

默认核对并复用 `build/native_hessian16/` 的已选产物,不重新量化或编译;缺少固化资源时停止.随后打包、上传并准备固定小子集参考,服务器不自动执行板端推理.登录开发板后在部署目录执行:

```bash
bash compile.sh
```

板端检查少量输入,显示输入、实际聊天模板及流式文本,示例最多生成 32 Token,可能仅为回答片段;输出位于 `examples/output/`.compile 子集精度和数值对比只在终端显示,不生成正式报告.本次启用配方后未重新生成示例或运行新的评价.

`deploy/evaluate.sh` 是独立正式评价入口,不重新转换或编译,结果只写入 `results/summary.json`.正式 MMLU 为固定 1000 题、57 学科、seed=17,WikiText2 为完整 test;本轮没有执行正式范围评价.使用与板端目录规范见 [Genio 720 文档](../../../../docs/genio_720.md).

## DLA 文件

- `1024c/MiniCPM5-2B_asym4W_sym16A_42layer_prefill_128t_1024c.dla`
- `1024c/MiniCPM5-2B_asym4W_sym16A_42layer_decode_1t_1024c.dla`

实际加载路径由 `manifest.json` 的 `model_files` 指定.两个 DLA、embedding、分词器和张量契约必须配套,不能混用不同配方的资源.
