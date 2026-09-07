#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="${MODEL_ROOT:-/workspace/models/perception/image_classification/vit_base_patch16_224}"
readonly NP_ROOT="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211"
readonly NCC_BIN="${NP_ROOT}/neuron_sdk/host/bin/ncc-tflite"
# MT8189 (Genio 720) 的 NPU 为 MDLA 5.3 且无 EDPA 硬件, 与 YOLOv5s 相同的
# 约束: 必须 mdla5.3 + suppress-output + disallow-bridge, 详见 YOLOv5s README。
readonly NCC_ARCH="${NCC_ARCH:-mdla5.3}"
readonly OUTPUT_DLA="${OUTPUT_DLA:-${MODEL_ROOT}/models/model_int8.dla}"

test -f "${MODEL_ROOT}/models/model_int8.tflite"
export LD_LIBRARY_PATH="${NP_ROOT}/neuron_sdk/host/lib:${LD_LIBRARY_PATH:-}"

echo "[BUILD] 使用 ${NCC_ARCH} 编译 ViT INT8 TFLite 为 DLA。"
if [[ "${NCC_CHECK_ONLY:-0}" == "1" ]]; then
    # 探针模式: 验证执行计划是否存在 MDLA 之外的桥接目标。
    "${NCC_BIN}" --arch="${NCC_ARCH}" --suppress-output --disallow-bridge \
        --show-exec-plan "${MODEL_ROOT}/models/model_int8.tflite" \
        -o "${OUTPUT_DLA}"
    echo "[OK] 无桥接验证通过: ${OUTPUT_DLA}"
    exit 0
fi

"${NCC_BIN}" --arch="${NCC_ARCH}" --suppress-output --disallow-bridge \
    "${MODEL_ROOT}/models/model_int8.tflite" \
    -o "${OUTPUT_DLA}"
if [[ "${OUTPUT_DLA}" == "${MODEL_ROOT}/models/model_int8.dla" ]]; then
    sha256sum "${OUTPUT_DLA}" >> "${MODEL_ROOT}/models/SHA256SUMS"
fi
echo "[OK] DLA: ${OUTPUT_DLA}"
