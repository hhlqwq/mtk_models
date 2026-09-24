#!/usr/bin/env bash

set -euo pipefail
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly CALIBRATION_DIR="${DEPTH_CALIBRATION_DIR:-/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017/images}"

test -s "${MODEL_ROOT}/original/depth_anything_v2_vits.pth"
test -d "${MODEL_ROOT}/original/upstream/.git"
test -d "${CALIBRATION_DIR}"
echo "[1/2] 导出固定 518x518 ONNX。"
python "${SCRIPT_DIR}/export_model.py" \
    --upstream "${MODEL_ROOT}/original/upstream" \
    --weights "${MODEL_ROOT}/original/depth_anything_v2_vits.pth" \
    --output "${MODEL_ROOT}/models/model_fp32.onnx"
echo "[2/2] 使用 16 张 COCO 图片执行 INT8 转换。"
python "${SCRIPT_DIR}/convert_int8.py" \
    --onnx "${MODEL_ROOT}/models/model_fp32.onnx" \
    --image-dir "${CALIBRATION_DIR}" --samples 16 \
    --output "${MODEL_ROOT}/models/model_int8.tflite"
echo "[OK] ONNX 和 INT8 TFLite 已生成。"
