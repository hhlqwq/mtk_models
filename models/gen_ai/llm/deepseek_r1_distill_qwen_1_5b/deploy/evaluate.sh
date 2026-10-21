#!/usr/bin/env bash
# 对已有模型运行生成性能评价,正式精度未实现时据实标记.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 MTK_EVAL_SCOPE=full
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
    echo "[错误] 用法: bash evaluate.sh,不接受参数." >&2
    exit 2
fi
export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
export MTK_RUN_DIR="/tmp/hailongcodex/$(date +%F)/deepseek_r1_distill_qwen_1_5b_evaluate_${EVAL_RUN_ID}"
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9 || { echo "[错误] 本模型已有运行任务." >&2; exit 2; }
    CONFIG="${SCRIPT_DIR}/config.json"
    echo "[评价 1/2] 固定生成样例、性能与峰值 RSS,不作为全量基准精度."
    python3 "${SCRIPT_DIR}/board/infer.py" --config "${CONFIG}" --run-id "${EVAL_RUN_ID}"
    echo "[评价 2/2] 写入唯一 summary.json."
    python3 "${SCRIPT_DIR}/board/summary.py" --config "${CONFIG}" --run-id "${EVAL_RUN_ID}"
else
    : "${MTK_LLM_PYTHON:?请先进入已有容器}"
    "${MTK_LLM_PYTHON}" "${SCRIPT_DIR}/host/evaluate_reference.py"
    echo "[提示] 当前固定语料参考不代表正式 MMLU/WikiText 全量精度."
fi
