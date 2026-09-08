#!/bin/sh
# 板端 COCO 评测推理循环: 对 inputs/ 下每个 INT8 bin 跑一次 neuronrt.
# 用法: board_eval_loop.sh <DLA> <inputs_dir> <outputs_dir>

set -eu

DLA="$1"
INPUTS="$2"
OUTPUTS="$3"
NEURONRT=/usr/sbin/neuronrt

mkdir -p "${OUTPUTS}"
count=0
total=$(find "${INPUTS}" -maxdepth 1 -type f -name '*.bin' | wc -l)
if [ "${total}" -eq 0 ]; then
    echo "未找到板端评测输入: ${INPUTS}" >&2
    exit 1
fi

for bin in "${INPUTS}"/*.bin; do
    stem=$(basename "${bin}" .bin)
    out="${OUTPUTS}/${stem}"
    done_file="${out}.done"
    if [ -f "${done_file}" ]; then
        for index in 0 1 2; do
            if [ ! -s "${out}_${index}.bin" ]; then
                echo "完成标记存在但输出缺失或为空: ${out}_${index}.bin" >&2
                exit 1
            fi
        done
        count=$((count + 1))
        continue
    fi
    rm -f "${out}_0.bin" "${out}_1.bin" "${out}_2.bin"
    "${NEURONRT}" -m hw -a "${DLA}" -i "${bin}" \
        -o "${out}_0.bin" -o "${out}_1.bin" -o "${out}_2.bin" \
        > /dev/null 2>&1
    for index in 0 1 2; do
        if [ ! -s "${out}_${index}.bin" ]; then
            echo "板端输出缺失或为空: ${out}_${index}.bin" >&2
            exit 1
        fi
    done
    touch "${done_file}"
    count=$((count + 1))
    if [ $((count % 50)) -eq 0 ]; then
        echo "[board] ${count}/${total} done"
    fi
done
echo "[board] 全部完成: ${count}/${total}"
