#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="/data/users/hailong.he/github/mtk_models/models/perception/object_detection/yolov5s"
readonly CALIBRATION_DIR="/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017/images"
readonly SOURCE_DIR="${MODEL_ROOT}/models/yolov5"
readonly SOURCE_ARCHIVE="${MODEL_ROOT}/models/yolov5-485da42.zip"
readonly SOURCE_ARCHIVE_ROOT="yolov5-485da42273839d20ea6bdaf142fd02c1027aba61"
readonly PATCH_ARCHIVE="${MODEL_ROOT}/models/model_conversion_YOLOv5s_example_20240916.zip"
readonly PATCH_DIR="${MODEL_ROOT}/models/mtk_patch"
readonly PATCH_FILE="${PATCH_DIR}/Fix_yolov5_mtk_tflite_issue.patch"

test -f "${MODEL_ROOT}/models/yolov5s.pt"
test -f "${SOURCE_ARCHIVE}"
test -f "${PATCH_ARCHIVE}"
test -d "${CALIBRATION_DIR}"

echo "[1/4] 展开本地源码并应用 MTK 补丁."
if [[ ! -d "${SOURCE_DIR}" ]]; then
    unzip -q "${SOURCE_ARCHIVE}" -d "${MODEL_ROOT}/models"
    mv "${MODEL_ROOT}/models/${SOURCE_ARCHIVE_ROOT}" "${SOURCE_DIR}"
fi
test -f "${SOURCE_DIR}/export.py"
mkdir -p "${PATCH_DIR}"
unzip -q -j -o "${PATCH_ARCHIVE}" -d "${PATCH_DIR}"
if git -C "${SOURCE_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "[INFO] MTK 补丁已应用."
else
    git -C "${SOURCE_DIR}" apply --check "${PATCH_FILE}"
    git -C "${SOURCE_DIR}" apply "${PATCH_FILE}"
fi

echo "[2/4] 验证镜像内预装工具链."
bash /opt/mtk-build/setup_container.sh

echo "[3/4] 导出 MTK 转换用 TorchScript 和 FP32 ONNX."
cd "${SOURCE_DIR}"
python export.py \
    --weights "${MODEL_ROOT}/models/yolov5s.pt" \
    --img-size 640 640 \
    --batch-size 1 \
    --device 0 \
    --include torchscript onnx
mv "${MODEL_ROOT}/models/yolov5s.onnx" "${MODEL_ROOT}/models/model_fp32.onnx"

echo "[4/4] 使用 MTK PyTorch Converter 执行 INT8 PTQ."
python "${MODEL_ROOT}/deploy/python/convert_int8.py" \
    --torchscript "${MODEL_ROOT}/models/yolov5s.torchscript" \
    --calibration-dir "${CALIBRATION_DIR}" \
    --output "${MODEL_ROOT}/models/model_int8.tflite"
sha256sum "${MODEL_ROOT}/models/yolov5s.pt" \
    "${MODEL_ROOT}/models/yolov5s.torchscript" \
    "${MODEL_ROOT}/models/model_fp32.onnx" \
    "${MODEL_ROOT}/models/model_int8.tflite" \
    > "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] YOLOv5s ONNX 和 INT8 TFLite 已生成."
