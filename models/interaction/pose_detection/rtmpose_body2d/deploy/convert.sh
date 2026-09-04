#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="/workspace/models/interaction/pose_detection/rtmpose_body2d"
readonly CALIBRATION_DIR="/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017/images"

test -f "${MODEL_ROOT}/models/model_fp32.onnx"
if [[ ! -d "${CALIBRATION_DIR}" ]]; then
    echo "[ERROR] 缺少 COCO 校准图片目录: ${CALIBRATION_DIR}" >&2
    exit 2
fi

python "${MODEL_ROOT}/deploy/convert_int8.py" \
    --onnx "${MODEL_ROOT}/models/model_fp32.onnx" \
    --calibration-dir "${CALIBRATION_DIR}" \
    --output "${MODEL_ROOT}/models/model_int8.tflite"
sha256sum "${MODEL_ROOT}/models/model_int8.tflite" \
    >> "${MODEL_ROOT}/models/SHA256SUMS"
