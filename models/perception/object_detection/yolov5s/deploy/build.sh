#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="${MODEL_ROOT:-/data/users/hailong.he/github/mtk_models/models/perception/object_detection/yolov5s}"
readonly NCC_BIN="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/bin/ncc-tflite"
readonly NCC_LIB="/opt/mtk/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211/neuron_sdk/host/lib"
# MT8189 (Genio 720) 的 NPU 为 MDLA 5.3, 板端 neuronrt 8.2.16 不支持 mdla3.0.
# --suppress-output: MT8189 无 EDPA 硬件, NCC 的 MDLA->Output 数据转换桥默认
# 落在 EDPA_1_2 上会导致加载失败; 抑制桥接后输出为 MDLA 原生 NCHW INT8,
# 行 stride 按 16 元素对齐 (见 postprocess_outputs.py 的 deinterleave).
readonly NCC_ARCH="${NCC_ARCH:-mdla5.3}"
readonly OUTPUT_DLA="${OUTPUT_DLA:-${MODEL_ROOT}/models/model_int8.dla}"

test -f "${MODEL_ROOT}/models/model_int8.tflite"
export LD_LIBRARY_PATH="${NCC_LIB}:${LD_LIBRARY_PATH:-}"

echo "[BUILD] 使用 ${NCC_ARCH} 编译 YOLOv5s INT8 TFLite 为 DLA."
if [[ "${NCC_CHECK_ONLY:-0}" == "1" ]]; then
    # 探针模式: 仅验证不存在 MDLA 到 Output 的桥接, 不产出正式 DLA.
    "${NCC_BIN}" --arch="${NCC_ARCH}" --suppress-output --disallow-bridge \
        --show-exec-plan "${MODEL_ROOT}/models/model_int8.tflite" -o "${OUTPUT_DLA}"
    echo "[OK] 无桥接验证通过: ${OUTPUT_DLA}"
    exit 0
fi

"${NCC_BIN}" --arch="${NCC_ARCH}" --suppress-output --disallow-bridge \
    "${MODEL_ROOT}/models/model_int8.tflite" \
    -o "${OUTPUT_DLA}"
if [[ "${OUTPUT_DLA}" == "${MODEL_ROOT}/models/model_int8.dla" ]]; then
    sha256sum "${MODEL_ROOT}/models/yolov5s.pt" \
        "${MODEL_ROOT}/models/yolov5s.torchscript" \
        "${MODEL_ROOT}/models/model_fp32.onnx" \
        "${MODEL_ROOT}/models/model_int8.tflite" \
        "${OUTPUT_DLA}" \
        > "${MODEL_ROOT}/models/SHA256SUMS"
fi
echo "[OK] DLA: ${OUTPUT_DLA}"
