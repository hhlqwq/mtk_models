#!/usr/bin/env bash
# 原生部署包独立评价,只发布一个 summary.json.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
    echo "[错误] 用法: bash evaluate.sh,不接受参数." >&2
    exit 2
fi
export MTK_EVAL_SCOPE=full PYTHONDONTWRITEBYTECODE=1
export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
export MTK_RUN_DIR="/tmp/hailongcodex/$(date +%F)/native_evaluate_${EVAL_RUN_ID}"
exec 9>"${ROOT}/.run.lock"
flock -n 9 || { echo "[错误] 本模型已有运行任务." >&2; exit 2; }
python3 "${ROOT}/scripts/evaluate_native.py" --root "${ROOT}" --run-id "${EVAL_RUN_ID}" --publish
