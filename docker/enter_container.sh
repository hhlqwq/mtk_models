#!/usr/bin/env bash

set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/tools/storage.sh"
readonly CONTAINER_NAME="${MTK_CONTAINER_NAME}"

# 依赖和环境在构建、创建阶段准备,此处只进入已运行容器的终端.
exec docker exec -it --workdir "${PROJECT_ROOT}" "${CONTAINER_NAME}" bash -l
