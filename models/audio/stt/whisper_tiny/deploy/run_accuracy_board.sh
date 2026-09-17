#!/usr/bin/env bash

set -euo pipefail

readonly DATASET="${DATASET:?请设置 DATASET=librispeech 或 aishell1}"
readonly RUN_ID="${RUN_ID:?请设置唯一 RUN_ID}"
readonly REPO_ROOT="${REPO_ROOT:-/data/users/hailong.he/github/mtk_models}"
readonly MODEL_ROOT="${REPO_ROOT}/models/audio/stt/whisper_tiny"
readonly EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/.eval/whisper_tiny/${RUN_ID}}"
readonly BOARD_HOST="${BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_ROOT="${BOARD_ROOT:-/root/hailong.he/whisper_tiny}"
readonly BOARD_EVAL_ROOT="${BOARD_EVAL_ROOT:-${BOARD_ROOT}/eval/${RUN_ID}}"
readonly BOARD_OUTPUT="${BOARD_EVAL_ROOT}/board_predictions.jsonl"
readonly LOCAL_OUTPUT="${EVAL_ROOT}/board"

test -f "${MODEL_ROOT}/deploy/inference_demo/whisper_board_eval"
test -f "${EVAL_ROOT}/board_manifest.tsv"
test -f "${EVAL_ROOT}/decode_config.txt"
test -d "${EVAL_ROOT}/mels"

echo "[1/5] 创建板端独立运行目录."
ssh -o BatchMode=yes "${BOARD_HOST}" \
    "mkdir -p '${BOARD_EVAL_ROOT}/mels' '${BOARD_EVAL_ROOT}/logs'"

echo "[2/5] 部署批量评测程序、清单和解码配置."
scp -p "${MODEL_ROOT}/deploy/inference_demo/whisper_board_eval" \
    "${BOARD_HOST}:${BOARD_ROOT}/whisper_board_eval"
scp -p "${EVAL_ROOT}/board_manifest.tsv" \
    "${EVAL_ROOT}/decode_config.txt" \
    "${BOARD_HOST}:${BOARD_EVAL_ROOT}/"
ssh -o BatchMode=yes "${BOARD_HOST}" \
    "chmod +x '${BOARD_ROOT}/whisper_board_eval'"

echo "[3/5] 增量传输 FP16 Mel,不删除板端已有文件."
if command -v rsync >/dev/null 2>&1 && \
        ssh -o BatchMode=yes "${BOARD_HOST}" "command -v rsync" \
        >/dev/null 2>&1; then
    rsync -a --info=progress2 "${EVAL_ROOT}/mels/" \
        "${BOARD_HOST}:${BOARD_EVAL_ROOT}/mels/"
else
    echo "[WARN] rsync 不可用,回退到 scp -r."
    scp -pr "${EVAL_ROOT}/mels/." \
        "${BOARD_HOST}:${BOARD_EVAL_ROOT}/mels/"
fi

echo "[4/5] 持久加载双 DLA 并断点续跑正式评测."
set +e
ssh -o BatchMode=yes "${BOARD_HOST}" \
    "/usr/bin/time -v -o '${BOARD_EVAL_ROOT}/logs/resource_usage.txt' \
    '${BOARD_ROOT}/whisper_board_eval' \
    '${BOARD_ROOT}/models/encoder_fp32.dla' \
    '${BOARD_ROOT}/models/decoder_step_fp32.dla' \
    '${BOARD_EVAL_ROOT}/board_manifest.tsv' \
    '${BOARD_EVAL_ROOT}/decode_config.txt' \
    '${BOARD_OUTPUT}' \
    > '${BOARD_EVAL_ROOT}/logs/board.log' 2>&1"
runtime_status=$?
set -e

echo "[5/5] 取回 JSONL 与原始日志."
mkdir -p "${LOCAL_OUTPUT}"
scp -p "${BOARD_HOST}:${BOARD_OUTPUT}" \
    "${BOARD_HOST}:${BOARD_EVAL_ROOT}/logs/resource_usage.txt" \
    "${BOARD_HOST}:${BOARD_EVAL_ROOT}/logs/board.log" \
    "${LOCAL_OUTPUT}/"
if [[ ${runtime_status} -ne 0 ]]; then
    echo "[ERROR] 板端评测返回 ${runtime_status},可重复执行本脚本断点续跑." >&2
    exit "${runtime_status}"
fi

echo "[OK] 板端结果: ${LOCAL_OUTPUT}/board_predictions.jsonl"
echo "[NEXT] 执行 evaluate_accuracy.py 汇总 ${DATASET} 指标."
