#!/usr/bin/env bash
# 离线导出与量化,仅在 89 的现有 G720 容器执行.
set -euo pipefail
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly WEIGHTS="${FASTSAM_WEIGHTS:-${MODEL_ROOT}/original/FastSAM-s.pt}"
: "${FASTSAM_WEIGHTS_SHA256:?请设置已核实的官方权重 SHA-256}"
: "${FASTSAM_IMAGE:?请设置现有样例图片绝对路径}"
: "${FASTSAM_CALIBRATION_DIR:?请设置校准图片目录}"
test -f "${WEIGHTS}"
test -f "${FASTSAM_IMAGE}"
test -d "${FASTSAM_CALIBRATION_DIR}"
python "${SCRIPT_DIR}/export_model.py" --weights "${WEIGHTS}" \
    --weights-sha256 "${FASTSAM_WEIGHTS_SHA256}" --image "${FASTSAM_IMAGE}" \
    --output-dir "${MODEL_ROOT}/models"
python "${SCRIPT_DIR}/convert_int8.py" --onnx "${MODEL_ROOT}/models/model_fp32.onnx" \
    --calibration-dir "${FASTSAM_CALIBRATION_DIR}" --samples "${FASTSAM_CALIBRATION_SAMPLES:-100}" \
    --output "${MODEL_ROOT}/models/model_int8.tflite"
sha256sum "${WEIGHTS}" "${MODEL_ROOT}/models/model_fp32.onnx" \
    "${MODEL_ROOT}/models/model_int8.tflite" > "${MODEL_ROOT}/models/SHA256SUMS"
