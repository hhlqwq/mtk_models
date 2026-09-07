#!/usr/bin/env bash

# 名称为兼容统一交付结构而保留: 实际只执行离线校验、解压和 SHA-256 记录,
# 不包含任何网络下载。vit-onnx-float.zip 由用户下载到本机后同步到 89。
set -euo pipefail

readonly MODEL_ROOT="/workspace/models/perception/image_classification/vit_base_patch16_224"
readonly EXPECTED_ZIP="${MODEL_ROOT}/original/vit-onnx-float.zip"
readonly USER_ZIP="${MODEL_ROOT}/models/vit-onnx-float.zip"

echo "[1/3] 定位用户放置的 Qualcomm ViT FP32 ONNX 归档。"
if [[ -f "${EXPECTED_ZIP}" ]]; then
    ARCHIVE="${EXPECTED_ZIP}"
elif [[ -f "${USER_ZIP}" ]]; then
    ARCHIVE="${USER_ZIP}"
    mkdir -p "${MODEL_ROOT}/original"
    mv "${ARCHIVE}" "${EXPECTED_ZIP}"
    ARCHIVE="${EXPECTED_ZIP}"
else
    echo "[ERROR] 未找到 vit-onnx-float.zip, 请按 README 先由用户下载并放置。" >&2
    exit 2
fi

echo "[2/3] 离线解压并提取 ONNX。"
rm -rf "${MODEL_ROOT}/original/exported"
mkdir -p "${MODEL_ROOT}/original/exported"
unzip -o -q "${ARCHIVE}" -d "${MODEL_ROOT}/original/exported"
readonly ONNX_PATH="$(find "${MODEL_ROOT}/original/exported" -type f -name '*.onnx' | head -n 1)"
test -n "${ONNX_PATH}"
cp "${ONNX_PATH}" "${MODEL_ROOT}/models/model_fp32.onnx"

echo "[3/3] 记录来源校验值。"
sha256sum "${ARCHIVE}" "${MODEL_ROOT}/models/model_fp32.onnx" \
    > "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] ViT FP32 ONNX 已准备: ${ONNX_PATH}"
