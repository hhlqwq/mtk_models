#!/usr/bin/env bash
# 仅为 Genio 720 编译,禁止将其他 SoC 的架构参数混入本模型.
set -euo pipefail
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly NCC_ROOT="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host"
test -f "${MODEL_ROOT}/models/model_int8.tflite"
test -f "${MODEL_ROOT}/models/model_int8.json"
export LD_LIBRARY_PATH="${NCC_ROOT}/lib:${LD_LIBRARY_PATH:-}"
echo "[BUILD] FastSAM-s INT8 -> MDLA 5.3,禁止硬件桥接."
"${NCC_ROOT}/bin/ncc-tflite" --arch=mdla5.3 --suppress-output --disallow-bridge \
    "${MODEL_ROOT}/models/model_int8.tflite" -o "${MODEL_ROOT}/models/model_int8.dla"
sha256sum "${MODEL_ROOT}/models/model_int8.dla" > "${MODEL_ROOT}/models/DLA_SHA256SUMS"
python "${SCRIPT_DIR}/finalize_build.py" --model-dir "${MODEL_ROOT}/models"
echo "[OK] 编译完成,尚不代表板端运行成功."
