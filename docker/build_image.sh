#!/usr/bin/env bash
set -euo pipefail
readonly PROJECT_ROOT="/data/users/hailong.he/github/mtk_models"
readonly SDK_DIR="/data/users/hailong.he/data/MTKG720/NeuroPilotSDK/neuropilot-sdk-basic-8.0.11-build20260211"
test -d "${SDK_DIR}/offline_tool"
cd "${PROJECT_ROOT}"
echo "[BUILD] 构建 Ubuntu 预装镜像, MTK SDK 使用服务器本地文件."
docker buildx build --load --progress=plain --network host \
    --build-context "sdk=${SDK_DIR}" \
    --tag hhl_g720_311:ubuntu22.04-np8.0.11 --file docker/Dockerfile docker
