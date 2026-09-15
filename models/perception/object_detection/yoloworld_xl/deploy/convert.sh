#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly SOURCE_MODEL="${MODEL_ROOT}/models/model_fp32.onnx"
readonly COMPATIBLE_MODEL="${MODEL_ROOT}/models/model_fp32_opset13.onnx"
readonly REPORT="${MODEL_ROOT}/docs/onnx_equivalence.json"

echo "[1/3] 检查官方模型文件。"
bash "${SCRIPT_DIR}/download_original.sh"

echo "[2/3] 生成 Neuron EP 兼容的 opset 13 模型。"
python3 "${SCRIPT_DIR}/prepare_onnx.py" \
    --input "${SOURCE_MODEL}" \
    --output "${COMPATIBLE_MODEL}"

echo "[3/3] 验证官方模型与兼容模型的 CPU 输出完全一致。"
python3 "${SCRIPT_DIR}/verify_onnx_equivalence.py" \
    --source "${SOURCE_MODEL}" \
    --converted "${COMPATIBLE_MODEL}" \
    --report "${REPORT}"

sha256sum "${SOURCE_MODEL}" "${COMPATIBLE_MODEL}" \
    > "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] YOLO-World XL ONNX 兼容准备完成。"
