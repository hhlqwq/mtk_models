# RTMPose 板端 Demo

Demo 使用 COCO `instances_val2017.json` 的 person 框模拟上游检测器输出,按 1.25 倍边距和
192:256 输入比例执行仿射裁剪.兼容模型输入为 NCHW BGR `[0,255]`,再按 TFLite 参数量化.

板端 `run_board.sh` 使用两张不同图片完成冒烟推理,随后执行 20 次预热和 100 次性能采样.
`postprocess_keypoints.py` 反量化 `pred_x`/`pred_y`,按 SimCC split ratio 2.0 解码 133 点并
映射回原图.如果两张不同图片产生完全一致的输出,流程会立即失败.
`compare_backends.py` 使用同一份量化输入比较 FP32 ONNX 和板端 NPU 输出,记录 logits
绝对误差、关键点 argmax 一致率及输入坐标系像素距离.

本 Demo 验证的是单人体框 RTMPose 模型.整图多人应用仍需串联人体检测器,两者延迟不得混写.
