#!/bin/sh

set -eu

readonly DEMO_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
readonly INPUT_DIR="${DEMO_DIR}/inputs"
readonly OUTPUT_DIR="${DEMO_DIR}/output"
readonly NEURONRT="/usr/sbin/neuronrt"
readonly DLA_FILE="${DEMO_DIR}/model_int8.dla"

mkdir -p "${OUTPUT_DIR}"
test -x "${NEURONRT}"
test -f "${DLA_FILE}"

echo "[1/4] 对两张独立人体裁剪输入执行板端冒烟推理."
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
    echo "[BOARD] ${count}/2 ${stem}"
done
test "${count}" -eq 2

echo "[2/4] 预热 20 次并连续推理 100 次."
first_input=$(find "${INPUT_DIR}" -maxdepth 1 -type f -name '*.bin' | sort | head -n 1)
"${NEURONRT}" -m hw -a "${DLA_FILE}" -i "${first_input}" \
    -o "${OUTPUT_DIR}/warmup_0.bin" \
    -o "${OUTPUT_DIR}/warmup_1.bin" -c 20 -b 100 -r turbo \
    -l performance --verbose >"${OUTPUT_DIR}/warmup.log" 2>&1
"${NEURONRT}" -m hw -a "${DLA_FILE}" -i "${first_input}" \
    -o "${OUTPUT_DIR}/benchmark_0.bin" \
    -o "${OUTPUT_DIR}/benchmark_1.bin" -c 100 -b 100 -r turbo \
    -l performance --verbose >"${OUTPUT_DIR}/benchmark.log" 2>&1

echo "[3/4] 采样峰值内存和单次进程端到端耗时."
peak=0
"${NEURONRT}" -m hw -a "${DLA_FILE}" -i "${first_input}" \
    -o "${OUTPUT_DIR}/memory_0.bin" \
    -o "${OUTPUT_DIR}/memory_1.bin" -c 100 -b 100 -r turbo \
    -l performance --verbose >"${OUTPUT_DIR}/memory_run.log" 2>&1 &
pid=$!
while kill -0 "${pid}" 2>/dev/null; do
    value=$(awk '/VmHWM/{print $2}' "/proc/${pid}/status" 2>/dev/null) || true
    [ -n "${value}" ] && [ "${value}" -gt "${peak}" ] && peak=${value}
done
wait "${pid}"
test "${peak}" -gt 0
echo "PeakRssKB=${peak}" >"${OUTPUT_DIR}/memory.txt"
start=$(date +%s%N)
"${NEURONRT}" -m hw -a "${DLA_FILE}" -i "${first_input}" \
    -o "${OUTPUT_DIR}/e2e_0.bin" \
    -o "${OUTPUT_DIR}/e2e_1.bin" >/dev/null 2>&1
end=$(date +%s%N)
echo "OneShotProcessMs=$(( (end - start) / 1000000 ))" \
    >>"${OUTPUT_DIR}/memory.txt"

echo "[4/4] 保存运行环境与产物校验信息."
"${NEURONRT}" -v >"${OUTPUT_DIR}/neuronrt_version.txt" 2>&1 || true
uname -a >"${OUTPUT_DIR}/system.txt"
sha256sum "${DLA_FILE}" "${first_input}" >"${OUTPUT_DIR}/SHA256SUMS"
echo "[OK] RTMPose 板端冒烟和性能测试完成."
