#!/usr/bin/env bash

set -euo pipefail

readonly PROJECT_ROOT="/data/users/hailong.he/github/mtk_models"
readonly SDK_ROOT="/data/users/hailong.he/data/MTKG720"
readonly DATASET_ROOT="/data/users/hailong.he/nas_smb/Datasets/open_source/raw"
readonly IMAGE_NAME="hhl_g720_311:np8.0.11"
readonly CONTAINER_NAME="hhl_g720_311"

echo "[1/4] 检查目录和同名容器。"
test -d "${PROJECT_ROOT}"
test -d "${SDK_ROOT}"
test -d "${DATASET_ROOT}"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    echo "[INFO] 容器 ${CONTAINER_NAME} 已存在，不重复创建。"
    docker start "${CONTAINER_NAME}" >/dev/null
else
    echo "[2/4] 构建 Python 3.11 基础镜像。"
    docker build --progress=plain --network host \
        --tag "${IMAGE_NAME}" --file "${PROJECT_ROOT}/docker/Dockerfile" \
        "${PROJECT_ROOT}"

    echo "[3/4] 创建 Genio 720 转换容器。"
    docker run --detach \
        --name "${CONTAINER_NAME}" \
        --hostname "${CONTAINER_NAME}" \
        --gpus all \
        --network host \
        --shm-size 8g \
        --env NVIDIA_VISIBLE_DEVICES=all \
        --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        --volume "${PROJECT_ROOT}:/workspace" \
        --volume "${SDK_ROOT}:/opt/mtk:ro" \
        --volume "${DATASET_ROOT}:${DATASET_ROOT}:ro" \
        "${IMAGE_NAME}" >/dev/null
fi

echo "[4/4] 安装并验证 MTK 工具环境。"
docker exec "${CONTAINER_NAME}" bash /workspace/docker/setup_container.sh
echo "[OK] 容器 ${CONTAINER_NAME} 已就绪。"
