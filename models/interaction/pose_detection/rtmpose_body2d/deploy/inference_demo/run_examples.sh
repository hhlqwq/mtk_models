#!/bin/sh

set -eu

DEMO_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
INPUT_DIR="${DEMO_DIR}/inputs"
OUTPUT_DIR="${DEMO_DIR}/output"
NEURONRT="/usr/sbin/neuronrt"
DLA_FILE="${DEMO_DIR}/model_int8.dla"

mkdir -p "${OUTPUT_DIR}"
test -x "${NEURONRT}"
test -f "${DLA_FILE}"
total=$(find "${INPUT_DIR}" -maxdepth 1 -type f -name '*.bin' | wc -l)
test "${total}" -eq 3

count=0
for input_file in "${INPUT_DIR}"/*.bin; do
    stem=$(basename "${input_file}" .bin)
    "${NEURONRT}" -m hw -a "${DLA_FILE}" -i "${input_file}" \
        -o "${OUTPUT_DIR}/${stem}_0.bin" \
        -o "${OUTPUT_DIR}/${stem}_1.bin" \
        >"${OUTPUT_DIR}/${stem}.log" 2>&1
    test -s "${OUTPUT_DIR}/${stem}_0.bin"
    test -s "${OUTPUT_DIR}/${stem}_1.bin"
    count=$((count + 1))
    echo "[BOARD] ${count}/${total} ${stem}"
done
echo "[OK] RTMPose 三张公开图片板端推理完成."
