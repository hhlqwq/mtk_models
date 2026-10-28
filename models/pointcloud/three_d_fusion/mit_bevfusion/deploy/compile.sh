#!/usr/bin/env bash
# 主机编译上传,板端用一帧真实输入检查 CPU + NPU 完整链路.
set -euo pipefail
export MTK_EVAL_SCOPE=subset
export MTK_OUTPUT_METRICS=1
export MTK_EVALUATION_WORK=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    export PYTHONDONTWRITEBYTECODE=1
    export OPENBLAS_NUM_THREADS=4
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9 || { echo "[错误] 本模型已有任务运行." >&2; exit 2; }
    python3 "${SCRIPT_DIR}/board/compile_accuracy.py"
    python3 "${SCRIPT_DIR}/board/check.py"
    echo "[完成] 子集精度对比和示例已完成,子集结果见 results/summary_sub.json."
else
    : "${MTK_MODELS_ROOT:?请先进入已有编译容器}"
    echo "[容器 1/3] 导出完整权重对应子图并编译 MDLA 5.3."
    python "${SCRIPT_DIR}/host/compile.py"
    echo "[容器 2/3] 使用已有 nuScenes 数据准备一帧输入."
    python "${SCRIPT_DIR}/host/prepare.py"
    echo "[容器 3/3] 上传完整 Python 部署包."
    python "${SCRIPT_DIR}/host/upload.py"
    python "${MTK_MODELS_ROOT}/tools/compile_accuracy.py" "${SCRIPT_DIR}"
fi
