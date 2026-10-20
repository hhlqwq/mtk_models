#!/usr/bin/env bash
# 在长时间全量测试前确认开发板与编译主机时钟一致,避免错误时间戳和 TLS 故障.

set -euo pipefail

: "${MTK_BOARD_HOST:?请先加载本地平台配置或设置 MTK_BOARD_HOST}"
readonly BOARD_HOST="${MTK_BOARD_HOST}"
readonly HOST_EPOCH="$(date -u +%s)"
readonly BOARD_EPOCH="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    "${BOARD_HOST}" 'date -u +%s')"

if [[ ! "${BOARD_EPOCH}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] 无法读取开发板的 UTC 时间." >&2
    exit 2
fi
delta=$((HOST_EPOCH - BOARD_EPOCH))
if (( delta < 0 )); then
    delta=$((-delta))
fi
if (( delta > 600 )); then
    echo "[ERROR] 开发板与编译主机时钟相差 ${delta} 秒,请先校准板端时间." >&2
    exit 2
fi
echo "[OK] 开发板与编译主机时钟相差 ${delta} 秒."
