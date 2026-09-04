#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="/workspace/models/perception/object_detection/yolov5s"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_DIR="${MTK_BOARD_ROOT:-/root/hailong.he}/yolov5s"

test -f "${MODEL_ROOT}/models/model_int8.dla"
echo "[1/2] 创建板端目录。"
ssh "${BOARD_HOST}" "mkdir -p '${BOARD_DIR}'"
echo "[2/2] 部署 DLA 和 Demo。"
scp "${MODEL_ROOT}/models/model_int8.dla" "${BOARD_HOST}:${BOARD_DIR}/"
scp -r "${MODEL_ROOT}/deploy/inference_demo" "${BOARD_HOST}:${BOARD_DIR}/"
echo "[OK] YOLOv5s 已部署到 ${BOARD_HOST}:${BOARD_DIR}。"
