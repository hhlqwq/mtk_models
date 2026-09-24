#!/bin/sh

set -eu
readonly RUN_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
readonly NEURONRT="/usr/sbin/neuronrt"

test -s "${RUN_DIR}/model_int8.dla"
test -x "${NEURONRT}"
mkdir -p "${RUN_DIR}/output"
for stem in image_1 image_2 image_3; do
    test -s "${RUN_DIR}/inputs/${stem}.bin"
    echo "[BOARD] 硬件推理 ${stem}。"
    "${NEURONRT}" -m hw -a "${RUN_DIR}/model_int8.dla" \
        -i "${RUN_DIR}/inputs/${stem}.bin" \
        -o "${RUN_DIR}/output/${stem}.bin" \
        > "${RUN_DIR}/output/${stem}.log" 2>&1
    test -s "${RUN_DIR}/output/${stem}.bin"
done
"${NEURONRT}" -v > "${RUN_DIR}/output/neuronrt_version.txt" 2>&1 || true
uname -a > "${RUN_DIR}/output/system.txt"
sha256sum "${RUN_DIR}/model_int8.dla" "${RUN_DIR}"/inputs/*.bin \
    "${RUN_DIR}"/output/image_*.bin > "${RUN_DIR}/output/SHA256SUMS"
echo "[OK] 三次板端硬件推理已完成。"
