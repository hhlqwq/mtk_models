#!/usr/bin/env bash
# 原生部署包独立评价,只发布一个 summary.json.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MTK_EVAL_SCOPE=full PYTHONDONTWRITEBYTECODE=1
exec 9>"${ROOT}/.run.lock"
flock -n 9 || { echo "[错误] 本模型已有运行任务." >&2; exit 2; }
python3 "${ROOT}/scripts/evaluate_native.py"
