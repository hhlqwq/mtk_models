#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly CALIBRATION_DIR="${CALIBRATION_DIR:-${MODEL_ROOT}/examples/input/calibration}"
readonly WEIGHTS="${MODEL_ROOT}/original/mobilefacenet.pt"
readonly EXPECTED_SHA256="90a00ba1d8b0b688af3deb731ed53dca582e6106805d1bc3cfdef55f570493f4"

test -s "${WEIGHTS}"
test -d "${CALIBRATION_DIR}"
bash /opt/mtk-build/setup_container.sh
echo "[1/2] 使用固定 PyTorch 权重自行导出 ONNX."
python "${SCRIPT_DIR}/export_onnx.py" \
    --weights "${WEIGHTS}" --sha256 "${EXPECTED_SHA256}" \
    --output "${MODEL_ROOT}/models/model_fp32.onnx"
echo "[2/2] 使用对齐人脸执行 INT8 PTQ."
python "${SCRIPT_DIR}/convert_int8.py" \
    --onnx "${MODEL_ROOT}/models/model_fp32.onnx" \
    --image-dir "${CALIBRATION_DIR}" \
    --output "${MODEL_ROOT}/models/model_int8.tflite"
sha256sum "${WEIGHTS}" "${MODEL_ROOT}/models/model_fp32.onnx" \
    "${MODEL_ROOT}/models/model_int8.tflite" \
    > "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] ONNX 和 INT8 TFLite 已生成."
