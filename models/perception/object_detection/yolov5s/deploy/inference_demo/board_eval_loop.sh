#!/bin/sh
# 板端 COCO 评测推理循环: 对 inputs/ 下每个 INT8 bin 跑一次 neuronrt。
# 用法: board_eval_loop.sh <DLA> <inputs_dir> <outputs_dir>

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
    out="${OUTPUTS}/${stem}"
    if [ -f "${out}_2.bin" ]; then
        count=$((count + 1))
        continue
    fi
    "${NEURONRT}" -m hw -a "${DLA}" -i "${bin}" \
        -o "${out}_0.bin" -o "${out}_1.bin" -o "${out}_2.bin" \
        > /dev/null 2>&1 || echo "FAIL ${stem}" >&2
    count=$((count + 1))
    if [ $((count % 50)) -eq 0 ]; then
        echo "[board] ${count}/${total} done"
    fi
done
echo "[board] 全部完成: ${count}/${total}"
