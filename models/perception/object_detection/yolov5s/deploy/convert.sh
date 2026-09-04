#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="/workspace/models/perception/object_detection/yolov5s"
readonly CALIBRATION_DIR="/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017/images"
readonly SOURCE_DIR="${MODEL_ROOT}/original/yolov5"

test -f "${MODEL_ROOT}/models/yolov5s.pt"
test -d "${CALIBRATION_DIR}"

echo "[1/3] 安装 YOLOv5 导出依赖。"
python -m pip install --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir -r "${SOURCE_DIR}/requirements.txt"

echo "[2/3] 导出 FP32 ONNX。"
cd "${SOURCE_DIR}"
python export.py \
    --weights "${MODEL_ROOT}/models/yolov5s.pt" \
    --imgsz 640 640 \
    --batch-size 1 \
    --include onnx
mv "${MODEL_ROOT}/models/yolov5s.onnx" "${MODEL_ROOT}/models/model_fp32.onnx"

echo "[3/3] 使用 MTK Converter 执行 INT8 PTQ。"
python "${MODEL_ROOT}/deploy/convert_int8.py" \
    --onnx "${MODEL_ROOT}/models/model_fp32.onnx" \
    --calibration-dir "${CALIBRATION_DIR}" \
    --output "${MODEL_ROOT}/models/model_int8.tflite"
sha256sum "${MODEL_ROOT}/models/model_fp32.onnx" \
    "${MODEL_ROOT}/models/model_int8.tflite" \
    >> "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] YOLOv5s ONNX 和 INT8 TFLite 已生成。"
