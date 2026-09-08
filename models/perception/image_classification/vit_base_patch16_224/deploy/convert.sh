#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="/workspace/models/perception/image_classification/vit_base_patch16_224"
# NAS 原始目录内离线解包的 ImageNet val 图片.
readonly CALIBRATION_DIR="/data/users/hailong.he/nas_smb/Datasets/open_source/raw/ILSVRC2012/val"

test -f "${MODEL_ROOT}/models/model_mtk_compatible.onnx"
if [[ ! -d "${CALIBRATION_DIR}" ]]; then
    echo "[ERROR] 缺少 ImageNet 校准图片目录: ${CALIBRATION_DIR}" >&2
    exit 2
fi

echo "[1/2] 验证镜像内预装工具链."
bash /opt/mtk-build/setup_container.sh

echo "[2/2] 使用 MTK ONNX Converter 执行 INT8 PTQ."
# --offset 1000: 校准取 val[1000:1100], 与精度评测子集 val[0:1000] 完全错开.
python "${MODEL_ROOT}/deploy/convert_int8.py" \
    --onnx "${MODEL_ROOT}/models/model_mtk_compatible.onnx" \
    --calibration-dir "${CALIBRATION_DIR}" \
    --offset 1000 \
    --output "${MODEL_ROOT}/models/model_int8.tflite"
sha256sum "${MODEL_ROOT}/models/model_int8.tflite" \
    >> "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] ViT INT8 TFLite 已生成."
