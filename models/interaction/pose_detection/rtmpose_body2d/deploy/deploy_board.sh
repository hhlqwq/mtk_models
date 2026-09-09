#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="${MODEL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_DIR="${MTK_BOARD_ROOT:-/root/hailong.he}/rtmpose_body2d"
readonly COCO_ROOT="${COCO_ROOT:-/data/users/hailong.he/nas_smb/Datasets/open_source/raw/coco/coco_val2017}"
readonly INPUT_DIR="${MODEL_ROOT}/examples/input/generated"
readonly OUTPUT_DIR="${MODEL_ROOT}/examples/output"
readonly METADATA="${INPUT_DIR}/metadata.json"
readonly -a SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

test -f "${MODEL_ROOT}/models/model_int8.dla"
test -f "${MODEL_ROOT}/models/model_int8.tflite"
test -d "${COCO_ROOT}/images"
test -f "${COCO_ROOT}/annotations/instances_val2017.json"

echo "[1/5] 从两个独立 COCO person 框生成量化输入."
rm -rf "${INPUT_DIR}"
mkdir -p "${INPUT_DIR}"
python "${MODEL_ROOT}/deploy/inference_demo/prepare_input.py" \
    --image-dir "${COCO_ROOT}/images" \
    --annotations "${COCO_ROOT}/annotations/instances_val2017.json" \
    --tflite "${MODEL_ROOT}/models/model_int8.tflite" \
    --output-dir "${INPUT_DIR}" \
    --metadata "${METADATA}" \
    --count 2

echo "[2/5] 创建干净的板端运行目录."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_DIR}' && rm -rf '${BOARD_DIR}/inputs' '${BOARD_DIR}/output'"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_DIR}/inputs' '${BOARD_DIR}/output'"

echo "[3/5] 部署 DLA、输入和运行脚本."
scp "${SSH_OPTIONS[@]}" "${MODEL_ROOT}/models/model_int8.dla" \
    "${BOARD_HOST}:${BOARD_DIR}/model_int8.dla"
scp "${SSH_OPTIONS[@]}" "${INPUT_DIR}"/*.bin \
    "${BOARD_HOST}:${BOARD_DIR}/inputs/"
scp "${SSH_OPTIONS[@]}" \
    "${MODEL_ROOT}/deploy/inference_demo/run_board.sh" \
    "${BOARD_HOST}:${BOARD_DIR}/run_board.sh"

echo "[4/5] 执行板端双图冒烟和性能测试并回收证据."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "chmod +x '${BOARD_DIR}/run_board.sh' && '${BOARD_DIR}/run_board.sh'"
rm -rf "${OUTPUT_DIR}/board_raw"
mkdir -p "${OUTPUT_DIR}/board_raw"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "tar -C '${BOARD_DIR}/output' -cf - ." \
    | tar -C "${OUTPUT_DIR}/board_raw" -xf -

echo "[5/5] 反量化 SimCC 输出并生成 133 点结果."
python "${MODEL_ROOT}/deploy/inference_demo/postprocess_keypoints.py" \
    --metadata "${METADATA}" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}/board_raw" \
    --result-dir "${OUTPUT_DIR}"
echo "[OK] RTMPose 已部署并验证: ${BOARD_HOST}:${BOARD_DIR}."
