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
    CONFIG="${SCRIPT_DIR}/config.json"
    echo "[板端 1/4] C++ SDK 推理,运行编号: ${EVAL_RUN_ID}."
    python3 "${SCRIPT_DIR}/board/infer.py" --config "${CONFIG}" --run-id "${EVAL_RUN_ID}"
    echo "[板端 2/4] 精度评测."
    python3 "${SCRIPT_DIR}/board/full_accuracy_board.py" --config "${CONFIG}" --run-id "${EVAL_RUN_ID}"
    echo "[板端 3/4] 生成效果示例."
    python3 "${SCRIPT_DIR}/board/render_examples.py" --config "${CONFIG}" --run-id "${EVAL_RUN_ID}"
    echo "[板端 4/4] 汇总指标与模型大小."
    python3 "${SCRIPT_DIR}/board/summarize_board_result.py" --config "${CONFIG}" --run-id "${EVAL_RUN_ID}"
    python3 "${SCRIPT_DIR}/board/model_size.py" --config "${CONFIG}" --run-id "${EVAL_RUN_ID}" --include "*.dla"
else
    CONFIG="${SCRIPT_DIR}/../config.yaml"
    : "${MTK_MODELS_ROOT:?请先通过 docker/enter_container.sh 进入容器}"
    echo "[容器 1/6] 导出 FP32 ONNX."
    python3 "${SCRIPT_DIR}/host/export_model.py" --config "${CONFIG}"
    echo "[容器 2/6] 校准与量化."
    python3 "${SCRIPT_DIR}/host/convert_int8.py" --config "${CONFIG}"
    echo "[容器 3/6] 编译 DLA."
    python3 "${SCRIPT_DIR}/host/compile_model.py" --config "${CONFIG}"
    echo "[容器 4/6] ONNX 精度评测."
    python3 "${SCRIPT_DIR}/host/evaluate_onnx.py" --config "${CONFIG}"
    echo "[容器 5/6] 交叉编译板端 C++ 程序."
    make -C "${SCRIPT_DIR}/board"
    echo "[容器 6/6] 上传到开发板."
    python3 "${SCRIPT_DIR}/host/upload.py" --config "${CONFIG}"
fi
