#!/usr/bin/env bash

set -euo pipefail

readonly CONTAINER_NAME="hhl_g720_8011"
readonly PROJECT_ROOT="/data/users/hailong.he/github/mtk_models"

docker start "${CONTAINER_NAME}" >/dev/null
exec docker exec -it \
    --env MTK_BOARD_HOST="root@192.168.0.92" \
    --env MTK_BOARD_ROOT="/root/hailong.he" \
    --workdir "${PROJECT_ROOT}" \
    "${CONTAINER_NAME}" bash
