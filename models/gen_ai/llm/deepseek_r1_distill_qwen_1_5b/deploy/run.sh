#!/usr/bin/env bash
# 先通过 docker/enter_container.sh 进入编译容器; 板端直接运行上传的同一入口.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 部署固定使用小批量验证,全量精度请运行独立的 evaluate.sh.
export MTK_EVAL_SCOPE=subset
if (( $# != 0 )); then
    echo "[错误] 用法: bash run.sh; 全量精度使用 bash evaluate.sh." >&2
    exit 2
fi
export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
if [[ "${MTK_EVAL_SCOPE}" == subset && "${EVAL_RUN_ID}" != *_test ]]; then
    export EVAL_RUN_ID="${EVAL_RUN_ID}_test"
fi

if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    python3 "${SCRIPT_DIR}/board/infer.py"
    python3 "${SCRIPT_DIR}/board/model_size.py" --config "${SCRIPT_DIR}/config.json" --run-id "${EVAL_RUN_ID}" --include "*.dla" --include embedding_fp16.bin
else
    : "${MTK_LLM_PYTHON:?请通过 docker/enter_container.sh 进入容器}"
    "${MTK_LLM_PYTHON}" "${SCRIPT_DIR}/host/export_model.py"
    "${MTK_LLM_PYTHON}" "${SCRIPT_DIR}/host/evaluate_reference.py"
    "${MTK_LLM_PYTHON}" "${SCRIPT_DIR}/host/convert_model.py"
    "${MTK_LLM_PYTHON}" "${SCRIPT_DIR}/host/compile_model.py"
    make -C "${SCRIPT_DIR}/board"
    python3 "${SCRIPT_DIR}/host/upload.py"
fi
