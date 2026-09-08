#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="/workspace/models/perception/object_detection/yolov5s"
readonly CALIBRATION_DIR="/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017/images"
readonly SOURCE_DIR="${MODEL_ROOT}/original/yolov5"
readonly CONSTRAINTS_FILE="${MODEL_ROOT}/deploy/constraints-py311.txt"

test -f "${MODEL_ROOT}/models/yolov5s.pt"
test -d "${CALIBRATION_DIR}"

echo "[1/3] 验证镜像内预装工具链."
bash /opt/mtk-build/setup_container.sh

echo "[2/3] 导出 MTK 转换用 TorchScript 和 FP32 ONNX."
cd "${SOURCE_DIR}"
python export.py \
    --weights "${MODEL_ROOT}/models/yolov5s.pt" \
    --img-size 640 640 \
    --batch-size 1 \
    --device 0 \
    --include torchscript onnx
mv "${MODEL_ROOT}/models/yolov5s.onnx" "${MODEL_ROOT}/models/model_fp32.onnx"

echo "[3/3] 使用 MTK PyTorch Converter 执行 INT8 PTQ."
python "${MODEL_ROOT}/deploy/convert_int8.py" \
    --torchscript "${MODEL_ROOT}/models/yolov5s.torchscript" \
    --calibration-dir "${CALIBRATION_DIR}" \
    --output "${MODEL_ROOT}/models/model_int8.tflite"
sha256sum "${MODEL_ROOT}/models/yolov5s.pt" \
    "${MODEL_ROOT}/models/yolov5s.torchscript" \
    "${MODEL_ROOT}/models/model_fp32.onnx" \
    "${MODEL_ROOT}/models/model_int8.tflite" \
    > "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] YOLOv5s ONNX 和 INT8 TFLite 已生成."
