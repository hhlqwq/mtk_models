#!/usr/bin/env bash
# 原生阶段含义保留,每个阶段对应独立 Python 文件.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MTK_GAI_PYTHON:?请通过 docker/enter_container.sh 进入容器}"
if (( $# > 1 )); then
    echo "[错误] 用法: bash compile_native.sh [prepare|calibrate|quantize|shape|compile|all]" >&2
    exit 2
fi
STAGE="${1:-all}"
case "${STAGE}" in
    all|prepare|calibrate|quantize|shape|compile) ;;
    *) echo "[错误] 不支持的原生转换阶段: ${STAGE}" >&2; exit 2 ;;
esac
if [[ "${STAGE}" == all || "${STAGE}" == prepare ]]; then
    "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/prepare_native.py"
fi
if [[ "${STAGE}" == all || "${STAGE}" == calibrate ]]; then
    "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/calibrate.py"
fi
if [[ "${STAGE}" == all || "${STAGE}" == quantize ]]; then
    "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/quantize_model.py"
fi
if [[ "${STAGE}" == all || "${STAGE}" == shape ]]; then
    "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/shape_model.py"
fi
if [[ "${STAGE}" == all || "${STAGE}" == compile ]]; then
    "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/compile_native.py"
fi
