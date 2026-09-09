#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="${MODEL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
readonly COCO_ROOT="${COCO_ROOT:-/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017}"
readonly CALIBRATION_DIR="${COCO_ROOT}/images"
readonly ANNOTATIONS="${COCO_ROOT}/annotations/instances_val2017.json"

test -f "${MODEL_ROOT}/models/model_fp32.onnx"
test -f "${MODEL_ROOT}/models/rtmpose_body2d.data"
test -d "${CALIBRATION_DIR}"
test -f "${ANNOTATIONS}"

echo "[1/2] 验证镜像内预装工具链."
bash /opt/mtk-build/setup_container.sh
echo "[2/2] 使用 COCO person 框执行 RTMPose INT8 PTQ."
python "${MODEL_ROOT}/deploy/convert_int8.py" \
    --onnx "${MODEL_ROOT}/models/model_fp32.onnx" \
    --image-dir "${CALIBRATION_DIR}" \
    --annotations "${ANNOTATIONS}" \
    --output "${MODEL_ROOT}/models/model_int8.tflite"
sha256sum "${MODEL_ROOT}/models/model_int8.tflite" \
    >> "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] RTMPose INT8 TFLite 已生成."
