#!/bin/sh
# 板端 ViT 评测推理循环: inputs/ 每个 bin 单次 neuronrt, 输出 <stem>_0.bin。

set -eu

DLA="$1"
INPUTS="$2"
OUTPUTS="$3"
NEURONRT=/usr/sbin/neuronrt

mkdir -p "${OUTPUTS}"
count=0
total=$(ls "${INPUTS}" | grep -c '\.bin$')
for bin in "${INPUTS}"/*.bin; do
    stem=$(basename "${bin}" .bin)
    out="${OUTPUTS}/${stem}_0.bin"
    if [ -f "${out}" ]; then
        count=$((count + 1))
        continue
    fi
    "${NEURONRT}" -m hw -a "${DLA}" -i "${bin}" -o "${out}" \
        > /dev/null 2>&1 || echo "FAIL ${stem}" >&2
    count=$((count + 1))
    if [ $((count % 100)) -eq 0 ]; then
        echo "[board] ${count}/${total} done"
    fi
done
echo "[board] 全部完成: ${count}/${total}"
