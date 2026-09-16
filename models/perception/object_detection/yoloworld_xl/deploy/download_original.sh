#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly MODEL_PATH="${MODEL_ROOT}/models/model_fp32.onnx"
readonly EXPECTED_SIZE="419032016"
readonly EXPECTED_SHA256="6d5b231425200f0426b73967c33a69ce5af83e993426d4fc64dc1709571d6174"

# 校验用户从 MediaTek 官方地址下载并离线放置的模型.
echo "[1/2] 检查官方模型大小."
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "[ERROR] 缺少 ${MODEL_PATH},请按 README 下载后放置." >&2
    exit 1
fi
actual_size="$(stat -c '%s' "${MODEL_PATH}")"
if [[ "${actual_size}" != "${EXPECTED_SIZE}" ]]; then
    echo "[ERROR] 文件大小不匹配: ${actual_size}." >&2
    exit 1
fi

echo "[2/2] 检查官方模型 SHA-256."
actual_sha256="$(sha256sum "${MODEL_PATH}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
    echo "[ERROR] SHA-256 不匹配: ${actual_sha256}." >&2
    exit 1
fi
echo "[OK] 官方 YOLO-World XL ONNX 校验通过."
