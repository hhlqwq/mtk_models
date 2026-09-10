#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="${MODEL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
readonly WEIGHTS_NAME="vit_b_16-c867db91.pth"
readonly EXPECTED_SHA256="c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9"

if [[ -f "${MODEL_ROOT}/original/${WEIGHTS_NAME}" ]]; then
    readonly WEIGHTS_PATH="${MODEL_ROOT}/original/${WEIGHTS_NAME}"
elif [[ -f "${MODEL_ROOT}/models/${WEIGHTS_NAME}" ]]; then
    readonly WEIGHTS_PATH="${MODEL_ROOT}/models/${WEIGHTS_NAME}"
else
    echo "[ERROR] 未找到 TorchVision 官方权重: ${WEIGHTS_NAME}" >&2
    echo "[NEXT] 请由用户下载后放入 original/ 或 models/." >&2
    exit 2
fi

echo "[1/3] 校验 TorchVision ViT-B/16 官方权重."
echo "${EXPECTED_SHA256}  ${WEIGHTS_PATH}" | sha256sum --check --status || {
    echo "[ERROR] ViT 权重 SHA-256 不匹配: ${WEIGHTS_PATH}" >&2
    exit 1
}

echo "[2/3] 从官方 PyTorch 权重自行导出 ONNX."
python "${SCRIPT_DIR}/export_onnx.py" \
    --weights "${WEIGHTS_PATH}" \
    --reference-output "${MODEL_ROOT}/models/model_fp32.onnx" \
    --compatible-output "${MODEL_ROOT}/models/model_mtk_compatible.onnx"
python "${SCRIPT_DIR}/verify_onnx_equivalence.py" \
    --reference "${MODEL_ROOT}/models/model_fp32.onnx" \
    --converted "${MODEL_ROOT}/models/model_mtk_compatible.onnx"

echo "[3/3] 记录本地权重和 ONNX 校验值."
sha256sum "${WEIGHTS_PATH}" \
    "${MODEL_ROOT}/models/model_fp32.onnx" \
    "${MODEL_ROOT}/models/model_mtk_compatible.onnx" \
    > "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] ViT 开源上游 ONNX 已离线准备."
