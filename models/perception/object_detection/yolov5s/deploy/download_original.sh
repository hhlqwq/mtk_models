#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="/workspace/models/perception/object_detection/yolov5s"
readonly SOURCE_DIR="${MODEL_ROOT}/original/yolov5"
readonly SOURCE_COMMIT="485da42"
readonly WEIGHT_URL="https://github.com/ultralytics/assets/releases/download/v7.0/yolov5s.pt"
readonly MTK_PATCH_URL="https://mediatek-aiot.s3.ap-southeast-1.amazonaws.com/aiot/download/model-zoo/scripts/model_conversion_YOLOv5s_example_20240916.zip"

echo "[1/4] 获取锁定版本的 YOLOv5 源码。"
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    git clone https://github.com/ultralytics/yolov5.git "${SOURCE_DIR}"
fi
git -C "${SOURCE_DIR}" fetch --all --tags
git -C "${SOURCE_DIR}" checkout --detach "${SOURCE_COMMIT}"

echo "[2/4] 下载 YOLOv5s 原始权重。"
curl --fail --location --progress-bar "${WEIGHT_URL}" \
    --output "${MODEL_ROOT}/models/yolov5s.pt"

echo "[3/4] 下载 MTK 官方转换补丁。"
curl --fail --location --progress-bar "${MTK_PATCH_URL}" \
    --output "${MODEL_ROOT}/original/mtk_yolov5_patch.zip"
unzip -o "${MODEL_ROOT}/original/mtk_yolov5_patch.zip" \
    -d "${MODEL_ROOT}/original/mtk_patch"

echo "[4/4] 记录来源校验值。"
sha256sum "${MODEL_ROOT}/models/yolov5s.pt" \
    > "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] YOLOv5s 来源文件已准备。"
