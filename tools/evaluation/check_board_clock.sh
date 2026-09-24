#!/usr/bin/env bash
# 在长时间全量测试前确认 92 时钟与 89 一致,避免错误时间戳和 TLS 故障.

set -euo pipefail

if [[ "${MTK_BOARD_OPEN_MODELS_ROOT:-/root/hailong.he/open_models}" != \
        "/root/hailong.he/open_models" ]] ||
        [[ "${MTK_BOARD_DATASETS_ROOT:-/root/hailong.he/datasets}" != \
        "/root/hailong.he/datasets" ]]; then
    echo "[ERROR] 全量评测固定使用 /root/hailong.he/open_models 和 /root/hailong.he/datasets." >&2
    exit 2
fi

readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly HOST_EPOCH="$(date -u +%s)"
readonly BOARD_EPOCH="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    "${BOARD_HOST}" 'date -u +%s')"

if [[ ! "${BOARD_EPOCH}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] 无法读取 92 的 UTC 时间." >&2
    exit 2
fi
delta=$((HOST_EPOCH - BOARD_EPOCH))
if (( delta < 0 )); then
    delta=$((-delta))
fi
if (( delta > 600 )); then
    echo "[ERROR] 92 与 89 的时钟相差 ${delta} 秒,请先校准板端时间." >&2
    exit 2
fi
echo "[OK] 92 时钟与 89 相差 ${delta} 秒."
