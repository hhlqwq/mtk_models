#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="${MODEL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
readonly ASSET_URL="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/rtmpose_body2d/releases/v0.61.0/rtmpose_body2d-onnx-float.zip"
readonly ARCHIVE_PATH="${MODEL_ROOT}/original/rtmpose_body2d-onnx-float.zip"
readonly LEGACY_ARCHIVE_PATH="${MODEL_ROOT}/models/rtmpose_body2d-onnx-float.zip"

mkdir -p "${MODEL_ROOT}/original" "${MODEL_ROOT}/models"
if [[ ! -f "${ARCHIVE_PATH}" && -f "${LEGACY_ARCHIVE_PATH}" ]]; then
    echo "[1/3] 复用现有 Qualcomm RTMPose 归档."
    cp "${LEGACY_ARCHIVE_PATH}" "${ARCHIVE_PATH}"
elif [[ ! -f "${ARCHIVE_PATH}" ]]; then
    echo "[1/3] 下载 Qualcomm RTMPose FP32 ONNX 归档."
    curl --fail --location --continue-at - --progress-bar \
        "${ASSET_URL}" --output "${ARCHIVE_PATH}"
else
    echo "[1/3] 复用 original 中的 Qualcomm RTMPose 归档."
fi
echo "[2/3] 解压模型."
rm -rf "${MODEL_ROOT}/original/exported"
mkdir -p "${MODEL_ROOT}/original/exported"
unzip -o "${ARCHIVE_PATH}" -d "${MODEL_ROOT}/original/exported"
readonly ONNX_PATH="$(find "${MODEL_ROOT}/original/exported" -type f -name '*.onnx' | head -n 1)"
readonly DATA_PATH="$(find "${MODEL_ROOT}/original/exported" -type f -name '*.data' | head -n 1)"
test -n "${ONNX_PATH}"
test -n "${DATA_PATH}"
cp "${ONNX_PATH}" "${MODEL_ROOT}/models/model_fp32.onnx"
cp "${DATA_PATH}" "${MODEL_ROOT}/models/$(basename "${DATA_PATH}")"
echo "[3/3] 记录来源校验值."
sha256sum "${ARCHIVE_PATH}" "${MODEL_ROOT}/models/model_fp32.onnx" \
    "${MODEL_ROOT}/models/$(basename "${DATA_PATH}")" \
    > "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] RTMPose FP32 ONNX 已准备."
