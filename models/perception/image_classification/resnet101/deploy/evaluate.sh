#!/usr/bin/env bash
# 独立评价已有模型,不重新导出、量化、编译或上传.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MTK_EVAL_SCOPE=full

if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    export PYTHONDONTWRITEBYTECODE=1
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9 || { echo "[错误] 本模型已有运行任务." >&2; exit 2; }
    echo "[板端 1/2] 全量 ImageNet 推理、必要预热与性能采集."
    python3 "${SCRIPT_DIR}/board/infer.py"
    echo "[板端 2/2] 核对覆盖与参考,发布唯一 summary.json."
    python3 "${SCRIPT_DIR}/board/evaluate.py"
else
    : "${MTK_MODELS_ROOT:?请先进入已有编译容器}"
    echo "[全量参考] 使用已有 FP32 ONNX 计算 Top-1/Top-5."
    python "${SCRIPT_DIR}/host/evaluate_onnx.py"
    echo "[下一步] 将参考单独放入板端 models/onnx_reference.json,再在板端执行 bash evaluate.sh."
fi
