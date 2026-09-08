#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="/workspace/models/interaction/pose_detection/rtmpose_body2d"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_DIR="${MTK_BOARD_ROOT:-/root/hailong.he}/rtmpose_body2d"

test -f "${MODEL_ROOT}/models/model_int8.dla"
ssh "${BOARD_HOST}" "mkdir -p '${BOARD_DIR}'"
scp "${MODEL_ROOT}/models/model_int8.dla" "${BOARD_HOST}:${BOARD_DIR}/"
scp -r "${MODEL_ROOT}/deploy/inference_demo" "${BOARD_HOST}:${BOARD_DIR}/"
echo "[OK] RTMPose 已部署到 ${BOARD_HOST}:${BOARD_DIR}."
