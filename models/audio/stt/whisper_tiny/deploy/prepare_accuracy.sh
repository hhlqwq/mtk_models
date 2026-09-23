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
readonly BOARD_MEL_ROOT="${BOARD_MEL_ROOT:-${MTK_BOARD_DATASETS_ROOT:-/root/hailong.he/datasets}/whisper_tiny/${RUN_ID}/mels}"
readonly ARCHIVE="${ARCHIVE:-}"
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
set +e
docker exec -e PYTHONUNBUFFERED=1 -w "${REPO_ROOT}" "${CONTAINER}" \
    python "${MODEL_ROOT}/deploy/prepare_accuracy_mels.py" \
    --source-manifest "${EVAL_ROOT}/source_manifest.jsonl" \
    --output-dir "${EVAL_ROOT}" \
    --weights "${MODEL_ROOT}/original/tiny.pt" \
    --language "${LANGUAGE}" \
    --board-mel-root "${BOARD_MEL_ROOT}" \
    --reference-device "${REFERENCE_DEVICE}" \
    --batch-size "${REFERENCE_BATCH_SIZE}"
prepare_status=$?
set -e
docker exec --user 0:0 "${CONTAINER}" \
    chown -R "${HOST_UID}:${HOST_GID}" "${EVAL_ROOT}"
if [[ ${prepare_status} -ne 0 ]]; then
    echo "[ERROR] Mel/框架基线准备返回 ${prepare_status},修复后可重复执行." >&2
    exit "${prepare_status}"
fi

echo "[3/3] 显示待传输文件规模."
du -sh "${EVAL_ROOT}/audio" "${EVAL_ROOT}/mels"
sha256sum "${EVAL_ROOT}/source_manifest.jsonl" \
    "${EVAL_ROOT}/board_manifest.tsv" \
    "${EVAL_ROOT}/decode_config.txt"
echo "[OK] 输入准备完成: ${EVAL_ROOT}"
