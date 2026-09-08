#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="/workspace/models/interaction/pose_detection/rtmpose_body2d"
readonly ASSET_URL="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/rtmpose_body2d/releases/v0.61.0/rtmpose_body2d-onnx-float.zip"
readonly ARCHIVE_PATH="${MODEL_ROOT}/original/rtmpose_body2d-onnx-float.zip"

echo "[1/3] 下载 Qualcomm RTMPose FP32 ONNX 归档."
curl --fail --location --progress-bar "${ASSET_URL}" --output "${ARCHIVE_PATH}"
echo "[2/3] 解压模型."
rm -rf "${MODEL_ROOT}/original/exported"
mkdir -p "${MODEL_ROOT}/original/exported"
unzip -o "${ARCHIVE_PATH}" -d "${MODEL_ROOT}/original/exported"
readonly ONNX_PATH="$(find "${MODEL_ROOT}/original/exported" -type f -name '*.onnx' | head -n 1)"
test -n "${ONNX_PATH}"
cp "${ONNX_PATH}" "${MODEL_ROOT}/models/model_fp32.onnx"
echo "[3/3] 记录来源校验值."
sha256sum "${ARCHIVE_PATH}" "${MODEL_ROOT}/models/model_fp32.onnx" \
    > "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] RTMPose FP32 ONNX 已准备."
