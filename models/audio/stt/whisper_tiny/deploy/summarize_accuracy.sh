#!/usr/bin/env bash

set -euo pipefail

readonly DATASET="${DATASET:?请设置 DATASET=librispeech 或 aishell1}"
readonly RUN_ID="${RUN_ID:?请设置唯一 RUN_ID}"
readonly REPO_ROOT="${REPO_ROOT:-/data/users/hailong.he/github/mtk_models}"
readonly MODEL_ROOT="${REPO_ROOT}/models/audio/stt/whisper_tiny"
readonly EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/.eval/whisper_tiny/${RUN_ID}}"
readonly CONTAINER="${MTK_CONTAINER:-hhl_g720_8011}"
readonly HOST_UID="$(id -u)"
readonly HOST_GID="$(id -g)"

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] 未找到 docker,请退出容器并在 Ubuntu89 宿主机运行本脚本." >&2
    exit 127
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null)" != "true" ]]; then
    echo "[ERROR] MTK 容器未运行或不存在: ${CONTAINER}." >&2
    exit 2
fi

set +e
docker exec -e PYTHONUNBUFFERED=1 -w "${REPO_ROOT}" "${CONTAINER}" \
    python "${MODEL_ROOT}/deploy/evaluate_accuracy.py" \
    --dataset "${DATASET}" \
    --source-manifest "${EVAL_ROOT}/source_manifest.jsonl" \
    --board-predictions "${EVAL_ROOT}/board/board_predictions.jsonl" \
    --reference-predictions "${EVAL_ROOT}/reference_predictions.jsonl" \
    --preprocess-metrics "${EVAL_ROOT}/preprocess_metrics.jsonl" \
    --output-dir "${EVAL_ROOT}/report"
evaluate_status=$?
set -e
if [[ -d "${EVAL_ROOT}/report" ]]; then
    docker exec --user 0:0 "${CONTAINER}" \
        chown -R "${HOST_UID}:${HOST_GID}" "${EVAL_ROOT}/report"
fi
if [[ ${evaluate_status} -ne 0 ]]; then
    echo "[ERROR] 指标汇总返回 ${evaluate_status},请检查报告中的缺失或失败样例." >&2
    exit "${evaluate_status}"
fi

sha256sum "${EVAL_ROOT}/report/summary.json" \
    "${EVAL_ROOT}/report/worst_samples.jsonl" \
    "${EVAL_ROOT}/report/report.md"
echo "[OK] 正式精度与性能报告: ${EVAL_ROOT}/report"
