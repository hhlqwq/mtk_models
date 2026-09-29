# 历史测试实现

本目录保留 89 端后处理的旧对照流程和单输入 `neuronrt` Demo，以便追溯历史结果。当前测试请按模型 README 在 89 宿主机使用 `deploy/test_board.sh`。

`accuracy_eval.sh` 仍依赖同目录的 `board_eval_loop.sh` 和仓库的 `tools/accuracy/yolov5s_val_coco.py`。这些历史文件不参与新的板端 C++ 测试，不应拿旧结果替代新镜像验证。
