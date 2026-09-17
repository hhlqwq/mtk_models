#!/usr/bin/env bash

set -euo pipefail

readonly DATASET="${DATASET:?请设置 DATASET=librispeech 或 aishell1}"
readonly RUN_ID="${RUN_ID:?请设置唯一 RUN_ID}"
readonly REPO_ROOT="${REPO_ROOT:-/data/users/hailong.he/github/mtk_models}"
readonly MODEL_ROOT="${REPO_ROOT}/models/audio/stt/whisper_tiny"
readonly EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/.eval/whisper_tiny/${RUN_ID}}"

python3 "${MODEL_ROOT}/deploy/evaluate_accuracy.py" \
    --dataset "${DATASET}" \
    --source-manifest "${EVAL_ROOT}/source_manifest.jsonl" \
    --board-predictions "${EVAL_ROOT}/board/board_predictions.jsonl" \
    --reference-predictions "${EVAL_ROOT}/reference_predictions.jsonl" \
    --preprocess-metrics "${EVAL_ROOT}/preprocess_metrics.jsonl" \
    --output-dir "${EVAL_ROOT}/report"

sha256sum "${EVAL_ROOT}/report/summary.json" \
    "${EVAL_ROOT}/report/worst_samples.jsonl" \
    "${EVAL_ROOT}/report/report.md"
echo "[OK] 正式精度与性能报告: ${EVAL_ROOT}/report"
