#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_ROOT="${MTK_BOARD_ROOT:-/root/hailong.he}"
readonly BOARD_MODEL_DIR="${BOARD_ROOT}/yoloworld_xl/model"
readonly BOARD_DEMO_DIR="${BOARD_ROOT}/yoloworld_xl/demo"
readonly MODEL_PATH="${MODEL_ROOT}/models/model_fp32_raw.onnx"
readonly SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

echo "[1/4] 检查本机兼容模型和板端脚本."
test -f "${MODEL_PATH}"
test -f "${SCRIPT_DIR}/inference_demo/run_board.py"
test -f "${SCRIPT_DIR}/inference_demo/yoloworld_utils.py"

echo "[2/4] 创建板端规范目录."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_MODEL_DIR}' '${BOARD_DEMO_DIR}/runtime' \
        '${BOARD_DEMO_DIR}/public/images' '${BOARD_ROOT}/yoloworld_xl/eval'"

echo "[3/4] 上传兼容 ONNX、运行脚本和公开样例."
readonly LOCAL_SHA256="$(sha256sum "${MODEL_PATH}" | awk '{print $1}')"
readonly EXISTING_SHA256_LINE="$(ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "if test -f '${BOARD_MODEL_DIR}/model_fp32_raw.onnx'; then \
        sha256sum '${BOARD_MODEL_DIR}/model_fp32_raw.onnx'; fi")"
readonly EXISTING_SHA256="${EXISTING_SHA256_LINE%% *}"
if [[ "${EXISTING_SHA256}" == "${LOCAL_SHA256}" ]]; then
    echo "[INFO] 板端已存在相同模型,跳过 400 MiB 重复上传."
else
    scp "${SSH_OPTIONS[@]}" "${MODEL_PATH}" \
        "${BOARD_HOST}:${BOARD_MODEL_DIR}/"
fi
scp "${SSH_OPTIONS[@]}" \
    "${SCRIPT_DIR}/inference_demo/run_board.py" \
    "${SCRIPT_DIR}/inference_demo/yoloworld_utils.py" \
    "${BOARD_HOST}:${BOARD_MODEL_DIR}/"
scp "${SSH_OPTIONS[@]}" "${MODEL_ROOT}"/examples/input/public/*.jpg \
    "${BOARD_HOST}:${BOARD_DEMO_DIR}/public/images/"

echo "[4/4] 核对板端模型哈希."
readonly BOARD_SHA256_LINE="$(ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "sha256sum '${BOARD_MODEL_DIR}/model_fp32_raw.onnx'")"
readonly BOARD_SHA256="${BOARD_SHA256_LINE%% *}"
if [[ "${LOCAL_SHA256}" != "${BOARD_SHA256}" ]]; then
    echo "[ERROR] 板端模型 SHA-256 不匹配." >&2
    exit 1
fi
echo "[OK] 板端部署完成: ${BOARD_MODEL_DIR}."
