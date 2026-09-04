#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="/workspace/models/perception/image_classification/vit_base_patch16_224"
readonly CALIBRATION_DIR="/data/users/hailong.he/nas_smb/Datasets/open_source/raw/ILSVRC2012/val"

test -f "${MODEL_ROOT}/models/model_fp32.onnx"
if [[ ! -d "${CALIBRATION_DIR}" ]]; then
    echo "[ERROR] 缺少 ImageNet 校准图片目录: ${CALIBRATION_DIR}" >&2
    exit 2
fi

python "${MODEL_ROOT}/deploy/convert_int8.py" \
    --onnx "${MODEL_ROOT}/models/model_fp32.onnx" \
    --calibration-dir "${CALIBRATION_DIR}" \
    --output "${MODEL_ROOT}/models/model_int8.tflite"
sha256sum "${MODEL_ROOT}/models/model_int8.tflite" \
    >> "${MODEL_ROOT}/models/SHA256SUMS"
