#!/usr/bin/env bash

set -euo pipefail

readonly MODEL_ROOT="/workspace/models/perception/object_detection/yolov5s"
readonly SOURCE_DIR="${MODEL_ROOT}/original/yolov5"
readonly SOURCE_ARCHIVE="${MODEL_ROOT}/original/yolov5-485da42.zip"
readonly SOURCE_ARCHIVE_ROOT="yolov5-485da42273839d20ea6bdaf142fd02c1027aba61"
readonly SOURCE_SHA256="ba30792a44660ae95bcc1e2fee1ca369b98871893894fc68ac2b69e26cdc7dae"
readonly PATCH_ARCHIVE="${MODEL_ROOT}/original/model_conversion_YOLOv5s_example_20240916.zip"
readonly PATCH_SHA256="8cb3ee3f7059a522e47fecd1be6983404e455e3f7069c19e9fe0a750df6a6fb7"
readonly PATCH_DIR="${MODEL_ROOT}/original/mtk_patch"
readonly PATCH_FILE="${PATCH_DIR}/Fix_yolov5_mtk_tflite_issue.patch"
readonly WEIGHT_FILE="${MODEL_ROOT}/models/yolov5s.pt"

# 检查文件 SHA-256,避免使用错误或不完整的离线输入.
check_sha256() {
    local file_path="$1"
    local expected_sha256="$2"
    local actual_sha256

    actual_sha256="$(sha256sum "${file_path}" | cut -d ' ' -f 1)"
    if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
        echo "[ERROR] SHA-256 不匹配: ${file_path}" >&2
        echo "[ERROR] 期望: ${expected_sha256}" >&2
        echo "[ERROR] 实际: ${actual_sha256}" >&2
        exit 1
    fi
}

echo "[1/5] 检查本机准备并经 GitHub 同步的小文件."
test -f "${SOURCE_ARCHIVE}"
test -f "${PATCH_ARCHIVE}"
check_sha256 "${SOURCE_ARCHIVE}" "${SOURCE_SHA256}"
check_sha256 "${PATCH_ARCHIVE}" "${PATCH_SHA256}"

echo "[2/5] 检查手动放置的 YOLOv5s 权重."
if [[ ! -f "${WEIGHT_FILE}" ]]; then
    echo "[ERROR] 缺少 ${WEIGHT_FILE}.请按 README 在本机下载后手动放置." >&2
    exit 1
fi

echo "[3/5] 离线展开锁定版本的 YOLOv5 源码."
if [[ ! -d "${SOURCE_DIR}" ]]; then
    unzip -q "${SOURCE_ARCHIVE}" -d "${MODEL_ROOT}/original"
    mv "${MODEL_ROOT}/original/${SOURCE_ARCHIVE_ROOT}" "${SOURCE_DIR}"
fi
test -f "${SOURCE_DIR}/export.py"

echo "[4/5] 离线展开并应用 MTK 官方补丁."
mkdir -p "${PATCH_DIR}"
unzip -q -j -o "${PATCH_ARCHIVE}" -d "${PATCH_DIR}"
if git -C "${SOURCE_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "[INFO] MTK 补丁已应用,跳过重复操作."
else
    git -C "${SOURCE_DIR}" apply --check "${PATCH_FILE}"
    git -C "${SOURCE_DIR}" apply "${PATCH_FILE}"
fi

echo "[5/5] 记录来源文件校验值."
sha256sum "${SOURCE_ARCHIVE}" "${PATCH_ARCHIVE}" "${WEIGHT_FILE}" \
    > "${MODEL_ROOT}/models/SHA256SUMS"
echo "[OK] YOLOv5s 离线来源文件已准备."
