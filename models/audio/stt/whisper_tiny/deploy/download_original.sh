#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="${MODEL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
readonly WEIGHTS_PATH="${MODEL_ROOT}/original/tiny.pt"
readonly EXPECTED_SHA256="65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9"

if [[ ! -f "${WEIGHTS_PATH}" ]]; then
    echo "[ERROR] 未找到 OpenAI 官方权重: ${WEIGHTS_PATH}" >&2
    echo "[NEXT] 请由用户下载 tiny.pt 后放入 original/,脚本不会联网下载." >&2
    exit 2
fi

echo "[1/5] 校验 OpenAI Whisper-Tiny 官方权重."
echo "${EXPECTED_SHA256}  ${WEIGHTS_PATH}" | sha256sum --check --status || {
    echo "[ERROR] tiny.pt SHA-256 不匹配." >&2
    exit 1
}

echo "[2/5] 导出固定 Shape Encoder 与 Decoder-Step ONNX."
python "${SCRIPT_DIR}/export_onnx.py" \
    --weights "${WEIGHTS_PATH}" \
    --output-dir "${MODEL_ROOT}/models"

echo "[3/5] 审计 ONNX 图的 Shape、dtype 和算子."
python "${SCRIPT_DIR}/inspect_onnx.py" \
    "${MODEL_ROOT}/models/encoder_fp32.onnx" \
    "${MODEL_ROOT}/models/decoder_step_fp32.onnx"

echo "[4/5] 验证原始模型、固定 Cache 改写与 ONNX 数值一致性."
python "${SCRIPT_DIR}/verify_onnx_equivalence.py" \
    --weights "${WEIGHTS_PATH}" \
    --encoder "${MODEL_ROOT}/models/encoder_fp32.onnx" \
    --decoder "${MODEL_ROOT}/models/decoder_step_fp32.onnx"

echo "[5/5] 记录权重与 ONNX SHA-256."
sha256sum "${WEIGHTS_PATH}" \
    "${MODEL_ROOT}/models/encoder_fp32.onnx" \
    "${MODEL_ROOT}/models/decoder_step_fp32.onnx" \
    > "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] Whisper-Tiny ONNX 已离线准备."
