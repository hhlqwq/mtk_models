#!/bin/sh

set -eu

readonly RUN_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
readonly MODEL_FILE="${RUN_DIR}/model_int8.dla"
readonly INPUT_DIR="${RUN_DIR}/inputs"
readonly OUTPUT_DIR="${RUN_DIR}/output"
readonly NEURONRT="/usr/sbin/neuronrt"

test -s "${MODEL_FILE}"
test -x "${NEURONRT}"
mkdir -p "${OUTPUT_DIR}"
for stem in face_1 face_2 face_3; do
    input="${INPUT_DIR}/${stem}.bin"
    test -s "${input}"
    echo "[BOARD] 推理 ${stem}."
    "${NEURONRT}" -m hw -a "${MODEL_FILE}" -i "${input}" \
        -o "${OUTPUT_DIR}/${stem}.bin" \
        > "${OUTPUT_DIR}/${stem}.log" 2>&1
    test -s "${OUTPUT_DIR}/${stem}.bin"
done
"${NEURONRT}" -v > "${OUTPUT_DIR}/neuronrt_version.txt" 2>&1 || true
uname -a > "${OUTPUT_DIR}/system.txt"
sha256sum "${MODEL_FILE}" "${INPUT_DIR}/face_1.bin" \
    "${INPUT_DIR}/face_2.bin" "${INPUT_DIR}/face_3.bin" \
    > "${OUTPUT_DIR}/SHA256SUMS"
echo "[OK] 三次 MobileFaceNet 板端推理已完成."
