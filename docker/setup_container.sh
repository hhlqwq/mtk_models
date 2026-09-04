#!/usr/bin/env bash

set -euo pipefail

readonly NP_ROOT="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211"
readonly CONVERTER_WHEEL="${NP_ROOT}/offline_tool/mtk_converter-8.16.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
readonly QUANTIZATION_WHEEL="${NP_ROOT}/offline_tool/mtk_quantization-8.2.1-py3-none-any.whl"
readonly MARKER_FILE="/usr/local/share/hhl_g720_311_setup_complete"
readonly PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
readonly LIBCXX_SOURCE="${NP_ROOT}/neuron_sdk/host/lib/libc++.so.1"
readonly LIBCXX_TARGET="/usr/local/lib/libc++.so.1"

echo "[SETUP] 固化容器 pip 镜像。"
python -m pip config --global set global.index-url "${PIP_INDEX_URL}"

echo "[SETUP] 配置 NCC 所需的 libc++.so.1。"
test -f "${LIBCXX_SOURCE}"
if [[ ! -e "${LIBCXX_TARGET}" ]]; then
    ln -s "${LIBCXX_SOURCE}" "${LIBCXX_TARGET}"
elif [[ "$(readlink -f "${LIBCXX_TARGET}")" != "${LIBCXX_SOURCE}" ]]; then
    echo "[ERROR] ${LIBCXX_TARGET} 已存在，但未指向当前 NeuroPilot SDK。" >&2
    exit 1
fi
ldconfig

if [[ ! -f "${MARKER_FILE}" ]]; then
    echo "[SETUP] 安装基础 Python 依赖。"
    python -m pip install --no-cache-dir -r /workspace/docker/requirements.txt

    echo "[SETUP] 安装 NeuroPilot 离线工具。"
    python -m pip install --no-cache-dir "${CONVERTER_WHEEL}" "${QUANTIZATION_WHEEL}"
else
    echo "[SETUP] 已检测到环境标记，跳过重复安装。"
fi

echo "[VERIFY] Python: $(python --version 2>&1)"
echo "[VERIFY] 检查 NVIDIA GPU 透传。"
nvidia-smi --query-gpu=index,name,driver_version,memory.total \
    --format=csv,noheader
python -c "import cv2, mtk_converter, numpy, onnx, onnxruntime; print('[VERIFY] Python 依赖导入成功。'); print('[VERIFY] mtk_converter:', mtk_converter.__version__)"
test -x "${NP_ROOT}/neuron_sdk/host/bin/ncc-tflite"
"${NP_ROOT}/neuron_sdk/host/bin/ncc-tflite" --version 2>&1 | head -n 5
touch "${MARKER_FILE}"
