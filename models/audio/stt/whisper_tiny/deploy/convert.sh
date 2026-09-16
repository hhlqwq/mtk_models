#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="${MODEL_ROOT:-/data/users/hailong.he/github/mtk_models/models/audio/stt/whisper_tiny}"
readonly SCRIPT_DIR="${MODEL_ROOT}/deploy"

for model_name in encoder decoder_step; do
    test -f "${MODEL_ROOT}/models/${model_name}_fp32.onnx"
done

echo "[1/3] 验证镜像内预装工具链."
bash /opt/mtk-build/setup_container.sh

echo "[2/3] 转换 Encoder FP32 ONNX."
python "${SCRIPT_DIR}/convert_fp32.py" \
    --onnx "${MODEL_ROOT}/models/encoder_fp32.onnx" \
    --output "${MODEL_ROOT}/models/encoder_fp32.tflite"

echo "[3/3] 转换固定 KV Cache Decoder-Step FP32 ONNX."
python "${SCRIPT_DIR}/convert_fp32.py" \
    --onnx "${MODEL_ROOT}/models/decoder_step_fp32.onnx" \
    --output "${MODEL_ROOT}/models/decoder_step_fp32.tflite"
sha256sum "${MODEL_ROOT}/models/encoder_fp32.tflite" \
    "${MODEL_ROOT}/models/decoder_step_fp32.tflite" \
    >> "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] Whisper-Tiny FP32 TFLite 已生成."
