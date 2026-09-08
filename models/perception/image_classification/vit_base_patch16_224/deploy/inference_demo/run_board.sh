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

echo "[1/4] 单次真实输入冒烟推理."
"${NEURONRT}" -m hw -a "${DLA_FILE}" -i "${INPUT_FILE}" \
    -o "${OUTPUT_DIR}/output_0.bin" >"${OUTPUT_DIR}/smoke.log" 2>&1

echo "[2/4] 预热 10 次 + 连续推理 100 次."
"${NEURONRT}" -m hw -a "${DLA_FILE}" -i "${INPUT_FILE}" \
    -o "${OUTPUT_DIR}/warmup_0.bin" \
    -c 10 -b 100 -r turbo -l performance --verbose \
    >"${OUTPUT_DIR}/warmup.log" 2>&1
"${NEURONRT}" -m hw -a "${DLA_FILE}" -i "${INPUT_FILE}" \
    -o "${OUTPUT_DIR}/benchmark_0.bin" \
    -c 100 -b 100 -r turbo -l performance --verbose \
    >"${OUTPUT_DIR}/benchmark.log" 2>&1

echo "[3/4] 采样峰值内存 (VmHWM) 和单次进程端到端耗时."
peak=0
"${NEURONRT}" -m hw -a "${DLA_FILE}" -i "${INPUT_FILE}" \
    -o "${OUTPUT_DIR}/mem_0.bin" -c 100 -b 100 -r turbo \
    -l performance --verbose >"${OUTPUT_DIR}/mem_run.log" 2>&1 &
pid=$!
while kill -0 "${pid}" 2>/dev/null; do
    v=$(awk '/VmHWM/{print $2}' "/proc/${pid}/status" 2>/dev/null) || true
    [ -n "${v}" ] && [ "${v}" -gt "${peak}" ] && peak=${v}
done
wait "${pid}"
[ "${peak}" -gt 0 ]
echo "PeakRssKB=${peak}" > "${OUTPUT_DIR}/memory.txt"

start=$(date +%s%N)
"${NEURONRT}" -m hw -a "${DLA_FILE}" -i "${INPUT_FILE}" \
    -o "${OUTPUT_DIR}/e2e_0.bin" >/dev/null 2>&1
end=$(date +%s%N)
echo "OneShotProcessMs=$(( (end - start) / 1000000 ))" >> "${OUTPUT_DIR}/memory.txt"

echo "[4/4] 保存版本与校验信息."
"${NEURONRT}" -v >"${OUTPUT_DIR}/neuronrt_version.txt" 2>&1 || true
uname -a >"${OUTPUT_DIR}/system.txt"
{
    echo "governors:"
    for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [ -f "${governor}" ] || continue
        printf '%s=' "${governor}"
        cat "${governor}"
    done
    echo "current_freq_khz:"
    for frequency in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
        [ -f "${frequency}" ] || continue
        printf '%s=' "${frequency}"
        cat "${frequency}"
    done
} >"${OUTPUT_DIR}/cpu_frequency.txt"
sha256sum "${DLA_FILE}" "${INPUT_FILE}" >"${OUTPUT_DIR}/SHA256SUMS"
echo "[OK] ViT 板端冒烟和性能测试完成."
