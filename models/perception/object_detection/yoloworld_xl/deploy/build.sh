#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[1/2] 生成并校验 Neuron EP 兼容 ONNX。"
bash "${SCRIPT_DIR}/convert.sh"

echo "[2/2] 检查板端执行入口。"
python3 -m py_compile \
    "${SCRIPT_DIR}/inference_demo/yoloworld_utils.py" \
    "${SCRIPT_DIR}/inference_demo/run_board.py"
echo "[OK] YOLO-World XL 在线部署产物准备完成，无离线 DLA 编译步骤。"
