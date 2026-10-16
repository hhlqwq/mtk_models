#!/usr/bin/env bash

set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/tools/storage.sh"
readonly CONTAINER_NAME="${MTK_CONTAINER_NAME}"

docker start "${CONTAINER_NAME}" >/dev/null
exec docker exec -it \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env MTK_MODEL_STORAGE_ROOT="${MTK_MODEL_STORAGE_ROOT}" \
    --env MTK_DATASET_ROOT="${MTK_DATASET_ROOT}" \
    --workdir "${PROJECT_ROOT}" \
    "${CONTAINER_NAME}" bash --rcfile "${PROJECT_ROOT}/docker/env.sh"
