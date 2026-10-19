#!/usr/bin/env bash

set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/tools/storage.sh"
readonly DATASET_ROOT="${MTK_DATASET_ROOT}"
readonly MODEL_STORAGE_ROOT="${MTK_MODEL_STORAGE_ROOT}"
readonly TOOLCHAIN_ROOT="${MTK_TOOLCHAIN_ROOT}"
readonly IMAGE_NAME="${MTK_G720_IMAGE:-openexplorer/ai_toolchain_ubuntu_22_g720_gpu:np8.0.11}"
readonly CONTAINER_NAME="${MTK_CONTAINER_NAME}"

echo "[1/4] 检查目录和同名容器."
test -d "${PROJECT_ROOT}"
test -d "${DATASET_ROOT}"
test -d "${MODEL_STORAGE_ROOT}"
test -d "${TOOLCHAIN_ROOT}"
# 先确认 NAS 实际挂载,防止失去挂载时将资源写入同名本地目录.
for STORAGE_ROOT in "${MODEL_STORAGE_ROOT}" "${DATASET_ROOT}"; do
    STORAGE_TYPE="$(findmnt -T "${STORAGE_ROOT}" -n -o FSTYPE)"
    case "${STORAGE_TYPE}" in
        cifs|nfs|nfs4) ;;
        *) echo "[ERROR] 模型和数据集目录必须位于已挂载的 NAS." >&2; exit 1 ;;
    esac
done
docker image inspect "${IMAGE_NAME}" >/dev/null

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    if [[ "$(docker inspect --format '{{.Image}}' "${CONTAINER_NAME}")" != "$(docker image inspect --format '{{.Id}}' "${IMAGE_NAME}")" ]]; then
        echo "[ERROR] 同名容器为旧镜像, 需要先安排迁移." >&2
        exit 1
    fi
    if ! docker inspect --format '{{range .Mounts}}{{println .Source "=>" .Destination}}{{end}}' \
        "${CONTAINER_NAME}" | grep -Fxq "${PROJECT_ROOT} => ${PROJECT_ROOT}"; then
        echo "[ERROR] 同名容器的项目挂载路径不一致,需要重新创建容器." >&2
        exit 1
    fi
    if [[ "$(docker inspect --format '{{.Config.WorkingDir}}' "${CONTAINER_NAME}")" != "${PROJECT_ROOT}" ]]; then
        echo "[ERROR] 同名容器的默认工作目录不一致,需要重新创建容器." >&2
        exit 1
    fi
    if ! docker inspect --format '{{range .Mounts}}{{println .Source "=>" .Destination .RW}}{{end}}' \
        "${CONTAINER_NAME}" | grep -Fxq "${MODEL_STORAGE_ROOT} => ${MODEL_STORAGE_ROOT} true"; then
        echo "[ERROR] 现有容器缺少模型存储读写挂载,请先保存环境并安排重建." >&2
        exit 1
    fi
    if ! docker inspect --format '{{range .Mounts}}{{println .Source "=>" .Destination .RW}}{{end}}' \
        "${CONTAINER_NAME}" | grep -Fxq "${DATASET_ROOT} => ${DATASET_ROOT} false"; then
        echo "[ERROR] 现有容器缺少数据集只读挂载,请安排重建." >&2
        exit 1
    fi
    if ! docker inspect --format '{{range .Mounts}}{{println .Source "=>" .Destination}}{{end}}' \
        "${CONTAINER_NAME}" | grep -Fxq "${TOOLCHAIN_ROOT} => ${TOOLCHAIN_ROOT}"; then
        echo "[ERROR] 现有容器缺少公共工具链挂载,请先保存环境并安排重建." >&2
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
        --env MTK_G720_CPP_TOOLCHAIN_ROOT="${MTK_G720_CPP_TOOLCHAIN_ROOT}" \
        --volume "${PROJECT_ROOT}:${PROJECT_ROOT}" \
        --volume "${DATASET_ROOT}:${DATASET_ROOT}:ro" \
        --volume "${MODEL_STORAGE_ROOT}:${MODEL_STORAGE_ROOT}" \
        --volume "${TOOLCHAIN_ROOT}:${TOOLCHAIN_ROOT}" \
        --workdir "${PROJECT_ROOT}" \
        "${IMAGE_NAME}" >/dev/null
fi

echo "[4/4] 设置项目访问权限,不运行模型编译或测试."
# GPU 透传可能附加宿主机组编号,补齐组名以消除交互终端的 groups 警告.
docker exec "${CONTAINER_NAME}" sh -c '
    for GROUP_ID in $(id -G); do
        if ! getent group "$GROUP_ID" >/dev/null; then
            groupadd --gid "$GROUP_ID" "host_$GROUP_ID"
        fi
    done'
docker exec "${CONTAINER_NAME}" git config --global --add safe.directory \
    "${PROJECT_ROOT}"
echo "[OK] 容器 ${CONTAINER_NAME} 已就绪."
