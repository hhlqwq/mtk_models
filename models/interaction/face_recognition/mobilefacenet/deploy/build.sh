#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly NP_ROOT="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211"
readonly NCC_BIN="${NP_ROOT}/neuron_sdk/host/bin/ncc-tflite"
readonly TFLITE="${MODEL_ROOT}/models/model_int8.tflite"
readonly DLA="${MODEL_ROOT}/models/model_int8.dla"

test -s "${TFLITE}"
export LD_LIBRARY_PATH="${NP_ROOT}/neuron_sdk/host/lib:${LD_LIBRARY_PATH:-}"
echo "[1/1] 编译 MobileFaceNet INT8 DLA, 禁止 CPU 桥接."
"${NCC_BIN}" --arch=mdla5.3 --suppress-output --disallow-bridge \
    "${TFLITE}" -o "${DLA}"
sha256sum "${MODEL_ROOT}/original/mobilefacenet.pt" \
    "${MODEL_ROOT}/models/model_fp32.onnx" "${TFLITE}" "${DLA}" \
    > "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] DLA: ${DLA}"
