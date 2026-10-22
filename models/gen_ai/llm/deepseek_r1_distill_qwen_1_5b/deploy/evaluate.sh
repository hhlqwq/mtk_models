#!/usr/bin/env bash
# 对已有模型运行生成性能评价,正式精度未实现时据实标记.
set -euo pipefail
export MTK_EVAL_SCOPE=full
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    export PYTHONDONTWRITEBYTECODE=1
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9 || { echo "[错误] 本模型已有运行任务." >&2; exit 2; }
    echo "[评价 1/2] 固定生成样例、性能与峰值 RSS,不作为全量基准精度."
    python3 "${SCRIPT_DIR}/board/infer.py"
    echo "[评价 2/2] 写入唯一 summary.json."
    python3 "${SCRIPT_DIR}/board/summary.py"
else
    : "${MTK_LLM_PYTHON:?请先进入已有容器}"
    "${MTK_LLM_PYTHON}" "${SCRIPT_DIR}/host/evaluate_reference.py"
    echo "[提示] 当前固定语料参考不代表正式 MMLU/WikiText 全量精度."
fi
