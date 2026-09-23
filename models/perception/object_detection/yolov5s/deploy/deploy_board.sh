#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_MODEL_ROOT="${MTK_BOARD_OPEN_MODELS_ROOT:-/root/hailong.he/open_models}/yolov5s"
readonly BOARD_MODEL_DIR="${BOARD_MODEL_ROOT}/models"
readonly BOARD_DIR="${BOARD_MODEL_ROOT}/demo/smoke"

test -f "${MODEL_ROOT}/models/model_int8.dla"
test -f "${MODEL_ROOT}/examples/input/input_int8.bin"
echo "[1/2] 创建板端目录."
ssh "${BOARD_HOST}" "mkdir -p '${BOARD_DIR}' '${BOARD_MODEL_DIR}'"
echo "[2/2] 部署 DLA 和 Demo."
scp "${MODEL_ROOT}/models/model_int8.dla" \
    "${BOARD_HOST}:${BOARD_MODEL_DIR}/model_int8.dla"
scp "${MODEL_ROOT}/examples/input/input_int8.bin" \
    "${BOARD_HOST}:${BOARD_DIR}/input_int8.bin"
scp "${MODEL_ROOT}/deploy/inference_demo/run_board.sh" \
    "${BOARD_HOST}:${BOARD_DIR}/run_board.sh"
ssh "${BOARD_HOST}" "chmod +x '${BOARD_DIR}/run_board.sh' && '${BOARD_DIR}/run_board.sh'"
mkdir -p "${MODEL_ROOT}/examples/output"
scp -r "${BOARD_HOST}:${BOARD_DIR}/output/." \
    "${MODEL_ROOT}/examples/output/"
echo "[OK] YOLOv5s 已完成板端推理,结果位于 examples/output."
