#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_ROOT="${MTK_BOARD_OPEN_MODELS_ROOT:-/root/hailong.he/open_models}/mobilefacenet"
readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
readonly BOARD_RUN_DIR="${BOARD_ROOT}/demo/smoke/${RUN_ID}"
readonly INPUT_DIR="${MODEL_ROOT}/examples/input/generated"
readonly OUTPUT_DIR="${MODEL_ROOT}/examples/output/runs/${RUN_ID}"
readonly DLA="${MODEL_ROOT}/models/model_int8.dla"
readonly TFLITE="${MODEL_ROOT}/models/model_int8.tflite"
readonly CALIBRATION_DIR="${CALIBRATION_DIR:-${MODEL_ROOT}/examples/input/calibration}"
readonly -a SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

test -s "${DLA}"
test -s "${TFLITE}"
test -d "${CALIBRATION_DIR}"
echo "[1/5] 从当前 TFLite 生成三份量化输入."
python "${SCRIPT_DIR}/prepare_input.py" \
    --tflite "${TFLITE}" \
    --image-dir "${CALIBRATION_DIR}" \
    --output-dir "${INPUT_DIR}"

echo "[2/5] 创建独立的板端冒烟目录: ${BOARD_RUN_DIR}."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "mkdir -p '${BOARD_RUN_DIR}/inputs' '${BOARD_RUN_DIR}/output'"

echo "[3/5] 传送 DLA、输入及运行脚本并执行板端推理."
scp "${SSH_OPTIONS[@]}" "${DLA}" \
    "${BOARD_HOST}:${BOARD_RUN_DIR}/model_int8.dla"
scp "${SSH_OPTIONS[@]}" \
    "${INPUT_DIR}/face_1.bin" "${INPUT_DIR}/face_2.bin" \
    "${INPUT_DIR}/face_3.bin" \
    "${BOARD_HOST}:${BOARD_RUN_DIR}/inputs/"
scp "${SSH_OPTIONS[@]}" "${SCRIPT_DIR}/run_board.sh" \
    "${BOARD_HOST}:${BOARD_RUN_DIR}/run_board.sh"
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" \
    "chmod +x '${BOARD_RUN_DIR}/run_board.sh' && '${BOARD_RUN_DIR}/run_board.sh'"

echo "[4/5] 回收板端原始输出和系统记录."
mkdir -p "${OUTPUT_DIR}/raw"
scp -r "${SSH_OPTIONS[@]}" "${BOARD_HOST}:${BOARD_RUN_DIR}/output/." \
    "${OUTPUT_DIR}/raw/"
cp "${INPUT_DIR}/metadata.json" "${OUTPUT_DIR}/metadata.json"
sha256sum "${DLA}" "${TFLITE}" "${INPUT_DIR}/face_1.bin" \
    "${INPUT_DIR}/face_2.bin" "${INPUT_DIR}/face_3.bin" \
    > "${OUTPUT_DIR}/host_inputs_sha256.txt"

echo "[5/5] 校验向量并写入冒烟结果."
python "${SCRIPT_DIR}/verify_board.py" \
    --metadata "${OUTPUT_DIR}/metadata.json" \
    --output-dir "${OUTPUT_DIR}/raw" \
    --result "${OUTPUT_DIR}/smoke_result.json"
echo "[OK] 运行目录: ${OUTPUT_DIR}"
