#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="/workspace/models/interaction/pose_detection/rtmpose_body2d"
readonly NP_ROOT="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211"

test -f "${MODEL_ROOT}/models/model_int8.tflite"
export LD_LIBRARY_PATH="${NP_ROOT}/neuron_sdk/host/lib:${LD_LIBRARY_PATH:-}"
"${NP_ROOT}/neuron_sdk/host/bin/ncc-tflite" --arch=mdla3.0 \
    "${MODEL_ROOT}/models/model_int8.tflite" \
    -o "${MODEL_ROOT}/models/model_int8.dla"
sha256sum "${MODEL_ROOT}/models/model_int8.dla" \
    >> "${MODEL_ROOT}/models/SHA256SUMS"
