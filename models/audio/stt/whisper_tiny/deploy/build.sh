#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="${MODEL_ROOT:-/data/users/hailong.he/github/mtk_models/models/audio/stt/whisper_tiny}"
readonly NP_ROOT="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211"
readonly NCC_BIN="${NP_ROOT}/neuron_sdk/host/bin/ncc-tflite"
readonly NCC_ARCH="${NCC_ARCH:-mdla5.3}"
readonly NCC_MODE="${NCC_MODE:-check}"

export LD_LIBRARY_PATH="${NP_ROOT}/neuron_sdk/host/lib:${LD_LIBRARY_PATH:-}"

build_model() {
    # 编译单个 TFLite,并保留检查或严格 NPU 两种模式.
    local model_name="$1"
    local input_path="${MODEL_ROOT}/models/${model_name}_fp32.tflite"
    local output_path="${MODEL_ROOT}/models/${model_name}_fp32.dla"
    test -f "${input_path}"
    echo "[BUILD] ${model_name}: mode=${NCC_MODE}, arch=${NCC_ARCH}."
    if [[ "${NCC_MODE}" == "check" ]]; then
        "${NCC_BIN}" --arch="${NCC_ARCH}" --show-exec-plan \
            "${input_path}" -o "${output_path}"
    elif [[ "${NCC_MODE}" == "strict" ]]; then
        "${NCC_BIN}" --arch="${NCC_ARCH}" --suppress-input --suppress-output \
            --disallow-bridge "${input_path}" -o "${output_path}"
    else
        echo "[ERROR] NCC_MODE 仅支持 check 或 strict." >&2
        return 2
    fi
}

build_model encoder
build_model decoder_step
sha256sum "${MODEL_ROOT}/models/encoder_fp32.dla" \
    "${MODEL_ROOT}/models/decoder_step_fp32.dla" \
    >> "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] Whisper-Tiny 双 DLA 编译完成."
