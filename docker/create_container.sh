#!/usr/bin/env bash

set -euo pipefail

readonly PROJECT_ROOT="/data/users/hailong.he/github/mtk_models"
readonly DATASET_ROOT="/data/users/hailong.he/nas_smb/Datasets/open_source/raw"
readonly IMAGE_NAME="hhl_g720_311:ubuntu22.04-np8.0.11"
readonly CONTAINER_NAME="hhl_g720_311"

echo "[1/4] 检查目录和同名容器."
test -d "${PROJECT_ROOT}"
test -d "${DATASET_ROOT}"
docker image inspect "${IMAGE_NAME}" >/dev/null

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    if [[ "$(docker inspect --format '{{.Image}}' "${CONTAINER_NAME}")" != "$(docker image inspect --format '{{.Id}}' "${IMAGE_NAME}")" ]]; then
        echo "[ERROR] 同名容器为旧镜像, 需要先安排迁移." >&2
        exit 1
    fi
    echo "[INFO] 容器 ${CONTAINER_NAME} 已存在,不重复创建."
    docker start "${CONTAINER_NAME}" >/dev/null
else
    echo "[2/4] 使用已经构建好的 Ubuntu MTK 镜像."

    echo "[3/4] 创建 Genio 720 转换容器."
    docker run --detach \
        --name "${CONTAINER_NAME}" \
        --hostname "${CONTAINER_NAME}" \
        --gpus all \
        --network host \
        --shm-size 8g \
        --env NVIDIA_VISIBLE_DEVICES=all \
        --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        --volume "${PROJECT_ROOT}:/workspace" \
        --volume "${DATASET_ROOT}:${DATASET_ROOT}:ro" \
        "${IMAGE_NAME}" >/dev/null
fi

echo "[4/4] 验证镜像内工具, 不执行安装."
docker exec "${CONTAINER_NAME}" bash /opt/mtk-build/setup_container.sh
echo "[OK] 容器 ${CONTAINER_NAME} 已就绪."
