# YOLOv5s 部署代码

顶层只保留用户入口：`prepare.sh` 在 89 的 Genio 720 容器内转换并编译模型；`test_board.sh smoke|full` 在 89 宿主机调用 92 板端；`cleanup_full_accuracy.sh` 在报告备份后单独清理。`convert.sh` 和 `build.sh` 是仓库注册规范要求的分阶段构建入口。

- `scripts/`：交叉编译辅助脚本。
- `python/`：量化、指标计算和示例绘制代码。
- `cpp/`：板端 C++ 推理源码。
- `../archive/legacy_deploy/`：历史对照流程，不参与当前测试。

具体命令与数据前提见上级 [README](../README.md)。
