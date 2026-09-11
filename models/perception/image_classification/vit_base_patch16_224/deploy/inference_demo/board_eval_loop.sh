#!/bin/sh
# 板端 ViT 评测推理循环: inputs/ 每个 bin 单次 neuronrt, 输出 <stem>_0.bin.

set -eu

DLA="$1"
INPUTS="$2"
OUTPUTS="$3"
PROGRESS_INTERVAL="${4:-100}"
NEURONRT=/usr/sbin/neuronrt

mkdir -p "${OUTPUTS}"
count=0
total=$(find "${INPUTS}" -maxdepth 1 -type f -name '*.bin' | wc -l)
[ "${total}" -gt 0 ]
for bin in "${INPUTS}"/*.bin; do
    stem=$(basename "${bin}" .bin)
    out="${OUTPUTS}/${stem}_0.bin"
    if [ -s "${out}" ]; then
        count=$((count + 1))
        continue
    fi
    "${NEURONRT}" -m hw -a "${DLA}" -i "${bin}" -o "${out}" \
        > "${OUTPUTS}/${stem}.log" 2>&1
    if [ ! -s "${out}" ]; then
        echo "FAIL ${stem}: neuronrt 未生成非空输出." >&2
        exit 1
    fi
    count=$((count + 1))
    if [ $((count % PROGRESS_INTERVAL)) -eq 0 ] || [ "${count}" -eq "${total}" ]; then
        echo "[BOARD] ${count}/${total} ${stem}"
    fi
done
[ "${count}" -eq "${total}" ]
echo "[board] 全部完成: ${count}/${total}"
