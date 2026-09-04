#!/usr/bin/env bash

set -euo pipefail

readonly NP_ROOT="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211"
readonly CONVERTER_WHEEL="${NP_ROOT}/offline_tool/mtk_converter-8.16.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
readonly QUANTIZATION_WHEEL="${NP_ROOT}/offline_tool/mtk_quantization-8.2.1-py3-none-any.whl"
readonly MARKER_FILE="/usr/local/share/hhl_g720_311_setup_complete"

if [[ ! -f "${MARKER_FILE}" ]]; then
    echo "[SETUP] 安装基础 Python 依赖。"
    python -m pip install --no-cache-dir -r /workspace/docker/requirements.txt

    echo "[SETUP] 安装 NeuroPilot 离线工具。"
    python -m pip install --no-cache-dir "${CONVERTER_WHEEL}" "${QUANTIZATION_WHEEL}"
    touch "${MARKER_FILE}"
else
    echo "[SETUP] 已检测到环境标记，跳过重复安装。"
fi

echo "[VERIFY] Python: $(python --version 2>&1)"
python -c "import mtk_converter; print('[VERIFY] mtk_converter:', mtk_converter.__version__)"
test -x "${NP_ROOT}/neuron_sdk/host/bin/ncc-tflite"
"${NP_ROOT}/neuron_sdk/host/bin/ncc-tflite" --version 2>&1 | head -n 5 || true
