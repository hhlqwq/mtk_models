#!/usr/bin/env bash

set -euo pipefail

readonly DATASET="${DATASET:?请设置 DATASET=librispeech 或 aishell1}"
readonly DATASET_ROOT="${DATASET_ROOT:?请设置已解压数据集根目录}"
readonly RUN_ID="${RUN_ID:?请设置唯一 RUN_ID}"
readonly REPO_ROOT="${REPO_ROOT:-/data/users/hailong.he/github/mtk_models}"
readonly MODEL_ROOT="${REPO_ROOT}/models/audio/stt/whisper_tiny"
readonly EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/.eval/whisper_tiny/${RUN_ID}}"
readonly CONTAINER="${MTK_CONTAINER:-hhl_g720_8011}"
readonly REFERENCE_DEVICE="${REFERENCE_DEVICE:-cuda}"
readonly REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-16}"
readonly BOARD_EVAL_ROOT="${BOARD_EVAL_ROOT:-/root/hailong.he/whisper_tiny/eval/${RUN_ID}}"
readonly ARCHIVE="${ARCHIVE:-}"

if [[ "${DATASET}" == "librispeech" ]]; then
    readonly LANGUAGE="en"
elif [[ "${DATASET}" == "aishell1" ]]; then
    readonly LANGUAGE="zh"
else
    echo "[ERROR] DATASET 仅支持 librispeech 或 aishell1." >&2
    exit 2
fi

mkdir -p "${EVAL_ROOT}"
echo "[1/3] 构建标准 WAV 与来源清单,不会修改原始数据."
manifest_args=(
    --dataset "${DATASET}"
    --dataset-root "${DATASET_ROOT}"
    --output-dir "${EVAL_ROOT}"
)
if [[ -n "${ARCHIVE}" ]]; then
    manifest_args+=(--archive "${ARCHIVE}")
fi
python3 "${MODEL_ROOT}/deploy/build_accuracy_manifest.py" \
    "${manifest_args[@]}"

echo "[2/3] 在既有 MTK 容器生成 FP16 Mel 和 OpenAI 框架基线."
docker exec -e PYTHONUNBUFFERED=1 -w "${REPO_ROOT}" "${CONTAINER}" \
    python "${MODEL_ROOT}/deploy/prepare_accuracy_mels.py" \
    --source-manifest "${EVAL_ROOT}/source_manifest.jsonl" \
    --output-dir "${EVAL_ROOT}" \
    --weights "${MODEL_ROOT}/original/tiny.pt" \
    --language "${LANGUAGE}" \
    --board-mel-root "${BOARD_EVAL_ROOT}/mels" \
    --reference-device "${REFERENCE_DEVICE}" \
    --batch-size "${REFERENCE_BATCH_SIZE}"

echo "[3/3] 显示待传输文件规模."
du -sh "${EVAL_ROOT}/audio" "${EVAL_ROOT}/mels"
sha256sum "${EVAL_ROOT}/source_manifest.jsonl" \
    "${EVAL_ROOT}/board_manifest.tsv" \
    "${EVAL_ROOT}/decode_config.txt"
echo "[OK] 输入准备完成: ${EVAL_ROOT}"
