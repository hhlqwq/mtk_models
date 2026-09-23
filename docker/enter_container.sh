#!/usr/bin/env bash

set -euo pipefail

readonly CONTAINER_NAME="hhl_g720_8011"
readonly PROJECT_ROOT="/data/users/hailong.he/github/mtk_models"

docker start "${CONTAINER_NAME}" >/dev/null
exec docker exec -it \
    --env MTK_BOARD_HOST="root@192.168.0.92" \
    --env MTK_BOARD_ROOT="/root/hailong.he" \
    --env MTK_BOARD_OPEN_MODELS_ROOT="/root/hailong.he/open_models" \
    --env MTK_BOARD_DATASETS_ROOT="/root/hailong.he/datasets" \
    --env MTK_BOARD_MODELZOO_ROOT="/root/hailong.he/MTK_G720_DLA" \
    --workdir "${PROJECT_ROOT}" \
    "${CONTAINER_NAME}" bash
