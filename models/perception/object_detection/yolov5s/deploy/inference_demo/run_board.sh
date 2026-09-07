#!/bin/sh

set -eu

readonly DEMO_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
readonly OUTPUT_DIR="${DEMO_DIR}/output"
readonly NEURONRT="/usr/sbin/neuronrt"
readonly DLA_FILE="${DEMO_DIR}/model_int8.dla"
readonly INPUT_FILE="${DEMO_DIR}/input_int8.bin"

mkdir -p "${OUTPUT_DIR}"
test -x "${NEURONRT}"
test -f "${DLA_FILE}"
test -f "${INPUT_FILE}"

echo "[1/3] 执行一次真实输入冒烟推理。"
"${NEURONRT}" -m hw -a "${DLA_FILE}" -d \
    -i "${INPUT_FILE}" \
    -o "${OUTPUT_DIR}/output_0.bin" \
    -o "${OUTPUT_DIR}/output_1.bin" \
    -o "${OUTPUT_DIR}/output_2.bin" \
    -b 100 -r turbo -l performance --verbose \
    >"${OUTPUT_DIR}/smoke.log" 2>&1

echo "[2/3] 预热 10 次。"
"${NEURONRT}" -m hw -a "${DLA_FILE}" -i "${INPUT_FILE}" \
    -o "${OUTPUT_DIR}/warmup_0.bin" \
    -o "${OUTPUT_DIR}/warmup_1.bin" \
    -o "${OUTPUT_DIR}/warmup_2.bin" \
    -c 10 -b 100 -r turbo -l performance \
    >"${OUTPUT_DIR}/warmup.log" 2>&1

echo "[3/3] 连续推理 100 次并记录资源。"
/usr/bin/time -v -o "${OUTPUT_DIR}/benchmark_resource.txt" \
    "${NEURONRT}" -m hw -a "${DLA_FILE}" -i "${INPUT_FILE}" \
    -o "${OUTPUT_DIR}/benchmark_0.bin" \
    -o "${OUTPUT_DIR}/benchmark_1.bin" \
    -o "${OUTPUT_DIR}/benchmark_2.bin" \
    -c 100 -b 100 -r turbo -l performance --verbose \
    >"${OUTPUT_DIR}/benchmark.log" 2>&1

"${NEURONRT}" -v >"${OUTPUT_DIR}/neuronrt_version.txt" 2>&1 || true
uname -a >"${OUTPUT_DIR}/system.txt"
sha256sum "${DLA_FILE}" "${INPUT_FILE}" >"${OUTPUT_DIR}/SHA256SUMS"
echo "[OK] 板端冒烟和性能测试完成。"
