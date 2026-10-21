#!/usr/bin/env bash
# 独立全量精度: 只读取已有模型和评价资源,不量化、编译或部署.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
    echo "[错误] 用法: bash evaluate.sh,不接受参数." >&2
    exit 2
fi
export MTK_EVAL_SCOPE=full
export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
if [[ "${EVAL_RUN_ID}" != *_full ]]; then
    export EVAL_RUN_ID="${EVAL_RUN_ID}_full"
fi
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9 || { echo "[错误] 本模型已有运行任务." >&2; exit 2; }
    CONFIG="${SCRIPT_DIR}/config.json"
    [[ -s "${SCRIPT_DIR}/models/fp32_reference.json" ]] || {
        echo "[错误] 缺少 models/fp32_reference.json,请单独准备全量协议参考." >&2
        exit 2
    }
    echo "[全量 1/3] 已有模型推理,运行编号: ${EVAL_RUN_ID}."
    python3 "${SCRIPT_DIR}/board/infer.py" --config "${CONFIG}" --run-id "${EVAL_RUN_ID}"
    echo "[全量 2/3] 计算精度并核对样本覆盖."
    python3 "${SCRIPT_DIR}/board/full_accuracy_board.py" --config "${CONFIG}" --run-id "${EVAL_RUN_ID}"
    echo "[全量 3/3] 保存精度汇总,保留预测和协议."
    python3 "${SCRIPT_DIR}/board/accuracy_summary.py" --config "${CONFIG}" --run-id "${EVAL_RUN_ID}"
else
    : "${MTK_MODELS_ROOT:?请先进入已有编译容器,仅对已有浮点模型准备评价参考}"
    CONFIG="${SCRIPT_DIR}/../config.yaml"
    echo "[全量参考] 仅推理已有浮点模型,不导出、量化、编译或上传."
    "${MTK_YAMNET_PYTHON:?缺少已准备的 TensorFlow 环境}" "${SCRIPT_DIR}/host/evaluate_onnx.py" --backend tensorflow --config "${CONFIG}"
    python3 "${SCRIPT_DIR}/host/evaluate_onnx.py" --backend onnx --config "${CONFIG}"
    echo "[下一步] 将上述全量参考和必要评价输入单独放入现有部署包对应位置,再在板端执行 bash evaluate.sh."
fi
