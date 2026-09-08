#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_DIR="${MTK_BOARD_ROOT:-/root/hailong.he}/vit_base_patch16_224"
readonly DEFAULT_IMAGE="/data/users/hailong.he/nas_smb/Datasets/open_source/raw/ILSVRC2012/val/ILSVRC2012_val_00000001.JPEG"
readonly DEMO_IMAGE="${VIT_DEMO_IMAGE:-${DEFAULT_IMAGE}}"
readonly INPUT_BIN="${MODEL_ROOT}/examples/input/input_int8.bin"
readonly INPUT_METADATA="${MODEL_ROOT}/examples/input/input_metadata.json"
readonly -a SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

test -f "${MODEL_ROOT}/models/model_int8.dla"
test -f "${MODEL_ROOT}/models/model_int8.tflite"
test -f "${DEMO_IMAGE}"

echo "[1/4] 按当前 TFLite 量化参数生成 Demo 输入."
python "${MODEL_ROOT}/deploy/inference_demo/prepare_input.py" \
    --image "${DEMO_IMAGE}" \
    --tflite "${MODEL_ROOT}/models/model_int8.tflite" \
    --output "${INPUT_BIN}" \
    --metadata "${INPUT_METADATA}"

echo "[2/4] 创建板端目录."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "mkdir -p '${BOARD_DIR}'"
echo "[3/4] 部署 DLA 和 Demo."
scp "${SSH_OPTIONS[@]}" "${MODEL_ROOT}/models/model_int8.dla" \
    "${BOARD_HOST}:${BOARD_DIR}/model_int8.dla"
scp "${SSH_OPTIONS[@]}" "${INPUT_BIN}" \
    "${BOARD_HOST}:${BOARD_DIR}/input_int8.bin"
scp "${SSH_OPTIONS[@]}" "${MODEL_ROOT}/deploy/inference_demo/run_board.sh" \
    "${BOARD_HOST}:${BOARD_DIR}/run_board.sh"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "chmod +x '${BOARD_DIR}/run_board.sh' && '${BOARD_DIR}/run_board.sh'"
mkdir -p "${MODEL_ROOT}/examples/output"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "tar -C '${BOARD_DIR}/output' -cf - ." \
    | tar -C "${MODEL_ROOT}/examples/output" -xf -
echo "[4/4] 反量化输出并生成 Top-5."
postprocess_args=(
    --metadata "${INPUT_METADATA}"
    --output-dir "${MODEL_ROOT}/examples/output"
    --result "${MODEL_ROOT}/examples/output/top5.json"
)
if [[ -f "${MODEL_ROOT}/original/labels.txt" ]]; then
    postprocess_args+=(--labels "${MODEL_ROOT}/original/labels.txt")
fi
python "${MODEL_ROOT}/deploy/inference_demo/postprocess_top5.py" \
    "${postprocess_args[@]}"
echo "[OK] ViT 已完成板端推理, 结果位于 examples/output."
