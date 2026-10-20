#!/usr/bin/env bash
set -euo pipefail
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/tools/storage.sh"
readonly SDK_DIR="${MTK_HOST_SDK_ROOT}"
test -d "${SDK_DIR}/offline_tool"
cd "${PROJECT_ROOT}"
echo "[BUILD] 构建 Ubuntu 预装镜像, MTK SDK 使用服务器本地文件."
docker buildx build --load --progress=plain --network host \
    --build-context "sdk=${SDK_DIR}" \
    --build-arg "MTK_SDK_ROOT=${SDK_DIR}" \
    --tag openexplorer/ai_toolchain_ubuntu_22_g720_gpu:np8.0.11 \
    --file docker/Dockerfile docker
