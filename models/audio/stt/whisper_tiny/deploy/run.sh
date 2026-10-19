#!/usr/bin/env bash
# 先通过 docker/enter_container.sh 进入编译容器; 板端直接运行上传的同一入口.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MTK_EVAL_TEST=0
if (( $# == 1 )) && [[ "$1" == "-test" ]]; then
    export MTK_EVAL_TEST=1
elif (( $# != 0 )); then
    echo "[错误] 用法: bash run.sh [-test]" >&2
    exit 2
fi
export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
if [[ "${MTK_EVAL_TEST}" == 1 && "${EVAL_RUN_ID}" != *_test ]]; then
    export EVAL_RUN_ID="${EVAL_RUN_ID}_test"
fi

if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    python3 "${SCRIPT_DIR}/board/infer.py"
    python3 "${SCRIPT_DIR}/board/render_examples.py" --config "${SCRIPT_DIR}/config.json" --run-id "${EVAL_RUN_ID}"
    python3 "${SCRIPT_DIR}/board/summarize_board_result.py" --config "${SCRIPT_DIR}/config.json" --run-id "${EVAL_RUN_ID}"
    python3 "${SCRIPT_DIR}/board/model_size.py" --config "${SCRIPT_DIR}/config.json" --run-id "${EVAL_RUN_ID}" --include "*.dla"
else
    : "${MTK_MODELS_ROOT:?请先通过 docker/enter_container.sh 进入容器}"
    python3 "${SCRIPT_DIR}/host/export_onnx.py"
    python3 "${SCRIPT_DIR}/host/convert_model.py"
    python3 "${SCRIPT_DIR}/host/compile_model.py"
    python3 "${SCRIPT_DIR}/host/evaluate_onnx.py"
    make -C "${SCRIPT_DIR}/board"
    python3 "${SCRIPT_DIR}/host/upload.py"
fi
