#!/usr/bin/env bash
# 从本地配置读取资源目录,不在 Git 文件中保存内部完整路径.
MTK_STORAGE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MTK_STORAGE_ARGUMENTS=()
if [[ "${1:-}" == "--config" ]]; then
    MTK_STORAGE_ARGUMENTS=(--model-config "${2:?请提供模型配置路径}")
elif [[ -n "${1:-}" ]]; then
    MTK_STORAGE_ARGUMENTS=(--model "$1")
fi
MTK_STORAGE_VALUES="$(PYTHONDONTWRITEBYTECODE=1 python3 "${MTK_STORAGE_REPO}/tools/storage.py" "${MTK_STORAGE_ARGUMENTS[@]}")" || return 1
eval "${MTK_STORAGE_VALUES}"
unset MTK_STORAGE_VALUES MTK_STORAGE_ARGUMENTS MTK_STORAGE_REPO
