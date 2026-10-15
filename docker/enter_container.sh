#!/usr/bin/env bash

set -euo pipefail

readonly CONTAINER_NAME="hhl_g720_8011"
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/tools/storage.sh"

docker start "${CONTAINER_NAME}" >/dev/null
exec docker exec -it \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env MTK_MODEL_STORAGE_ROOT="${MTK_MODEL_STORAGE_ROOT}" \
    --env MTK_DATASET_ROOT="${MTK_DATASET_ROOT}" \
    --env MTK_BOARD_HOST="root@192.168.0.92" \
    --env MTK_BOARD_ROOT="/root/hailong.he" \
    --env MTK_BOARD_OPEN_MODELS_ROOT="/root/hailong.he/open_models" \
    --env MTK_BOARD_DATASETS_ROOT="/root/hailong.he/datasets" \
    --env MTK_BOARD_MODELZOO_ROOT="/root/hailong.he/MTK_G720_DLA" \
    --workdir "${PROJECT_ROOT}" \
    "${CONTAINER_NAME}" bash
