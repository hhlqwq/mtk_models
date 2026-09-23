#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly ROOTFS_IMAGE="${1:-}"
readonly BOARD_HOST="${MTK_BOARD_HOST:-root@192.168.0.92}"
readonly BOARD_MODEL_ROOT="${MTK_BOARD_OPEN_MODELS_ROOT:-/root/hailong.he/open_models}/yoloworld_xl"
readonly BOARD_RUNTIME_DIR="${BOARD_MODEL_ROOT}/runtime_v26"
readonly TEMP_ROOT="/tmp/hailongcodex/$(date +%Y%m%d)/yoloworld_runtime"
readonly ROOTFS_SHA256="ce857239c548dd66c65cfef49f809582c107405c597c6773f64115fa03650c9d"
readonly ADAPTER_NAME="libneuronusdk_adapter.mtk.so.8.2.16"
readonly RUNTIME_NAME="libneuronusdk_runtime.mtk.so.8.2.16"
readonly ADAPTER_SHA256="225f70c7fc5fb6fefa1ef8b662df7036ed36431b57252379ce0dd4b9e2210bcb"
readonly RUNTIME_SHA256="38d74829a50f802eba0539d9c37273e7743e60d692b2af3fb6054805d08a85de"
readonly SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

if [[ -z "${ROOTFS_IMAGE}" ]]; then
    echo "用法: ${SCRIPT_NAME} <rity-v26.0-rootfs.ext4.img>" >&2
    exit 2
fi
if [[ ! -f "${ROOTFS_IMAGE}" ]]; then
    echo "[ERROR] rootfs 镜像不存在: ${ROOTFS_IMAGE}." >&2
    exit 1
fi

echo "[1/5] 校验官方 v26.0 rootfs SHA-256."
echo "${ROOTFS_SHA256}  ${ROOTFS_IMAGE}" | sha256sum -c -

echo "[2/5] 从官方 v26.0 rootfs 提取 Neuron 8.2.16 运行库."
mkdir -p "${TEMP_ROOT}"
debugfs -R "dump /usr/lib/${ADAPTER_NAME} ${TEMP_ROOT}/${ADAPTER_NAME}" \
    "${ROOTFS_IMAGE}" >/dev/null
debugfs -R "dump /usr/lib/${RUNTIME_NAME} ${TEMP_ROOT}/${RUNTIME_NAME}" \
    "${ROOTFS_IMAGE}" >/dev/null

echo "[3/5] 校验官方运行库 SHA-256."
echo "${ADAPTER_SHA256}  ${TEMP_ROOT}/${ADAPTER_NAME}" | sha256sum -c -
echo "${RUNTIME_SHA256}  ${TEMP_ROOT}/${RUNTIME_NAME}" | sha256sum -c -

echo "[4/5] 上传到板端模型专用隔离目录."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" "mkdir -p '${BOARD_RUNTIME_DIR}'"
scp "${SSH_OPTIONS[@]}" \
    "${TEMP_ROOT}/${ADAPTER_NAME}" \
    "${TEMP_ROOT}/${RUNTIME_NAME}" \
    "${BOARD_HOST}:${BOARD_RUNTIME_DIR}/"

echo "[5/5] 创建 SONAME 链接并执行板端哈希复核."
ssh "${SSH_OPTIONS[@]}" "${BOARD_HOST}" bash -s -- \
    "${BOARD_RUNTIME_DIR}" "${ADAPTER_NAME}" "${RUNTIME_NAME}" \
    "${ADAPTER_SHA256}" "${RUNTIME_SHA256}" <<'BOARD_STAGE'
set -euo pipefail
readonly runtime_dir="$1"
readonly adapter_name="$2"
readonly runtime_name="$3"
readonly adapter_sha256="$4"
readonly runtime_sha256="$5"
cd "${runtime_dir}"
echo "${adapter_sha256}  ${adapter_name}" | sha256sum -c -
echo "${runtime_sha256}  ${runtime_name}" | sha256sum -c -
ln -sfn "${adapter_name}" libneuronusdk_adapter.mtk.so.8
ln -sfn libneuronusdk_adapter.mtk.so.8 libneuronusdk_adapter.mtk.so
ln -sfn "${runtime_name}" libneuronusdk_runtime.mtk.so.8
ln -sfn libneuronusdk_runtime.mtk.so.8 libneuronusdk_runtime.mtk.so
BOARD_STAGE

echo "[OK] 隔离运行库已部署: ${BOARD_RUNTIME_DIR}."
echo "[NEXT] 运行时设置 MTK_NEURON_RUNTIME_DIR=${BOARD_RUNTIME_DIR}."
