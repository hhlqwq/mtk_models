#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="/workspace/models/perception/object_detection/yolov5s"
readonly NCC_BIN="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/bin/ncc-tflite"
readonly NCC_LIB="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/lib"

test -f "${MODEL_ROOT}/models/model_int8.tflite"
export LD_LIBRARY_PATH="${NCC_LIB}:${LD_LIBRARY_PATH:-}"

echo "[BUILD] 编译 YOLOv5s INT8 TFLite 为 DLA。"
"${NCC_BIN}" --arch=mdla3.0 \
    "${MODEL_ROOT}/models/model_int8.tflite" \
    -o "${MODEL_ROOT}/models/model_int8.dla"
sha256sum "${MODEL_ROOT}/models/model_int8.dla" \
    >> "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] DLA: ${MODEL_ROOT}/models/model_int8.dla"
