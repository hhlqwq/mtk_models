#!/usr/bin/env bash
# 编译、部署与少量展示示例;正式指标只由 evaluate.sh 生成.
set -euo pipefail
export MTK_EVAL_SCOPE=subset
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    export PYTHONDONTWRITEBYTECODE=1
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9 || { echo "[错误] 本模型已有运行任务." >&2; exit 2; }
    echo "[板端 1/2] 少量输入检查与硬件推理."
    python3 "${SCRIPT_DIR}/board/infer.py"
    echo "[板端 2/2] 生成文本示例已保存至 examples/output/."
    echo "[编译完成] 通路和示例已准备,未生成精度、性能或 RSS 报告."
else
    : "${MTK_LLM_PYTHON:?请通过 docker/enter_container.sh 进入容器}"
    echo "[容器 1/6] 导出浮点模型."
    "${MTK_LLM_PYTHON}" "${SCRIPT_DIR}/host/export_model.py"
    echo "[容器 2/6] 准备少量示例输入."
    "${MTK_LLM_PYTHON}" "${SCRIPT_DIR}/host/evaluate_reference.py" --examples-only
    echo "[容器 3/6] 校准与转换."
    "${MTK_LLM_PYTHON}" "${SCRIPT_DIR}/host/convert_model.py"
    echo "[容器 4/6] 编译 DLA."
    "${MTK_LLM_PYTHON}" "${SCRIPT_DIR}/host/compile_model.py"
    echo "[容器 5/6] 交叉编译板端桥接库."
    make -C "${SCRIPT_DIR}/board"
    echo "[容器 6/6] 上传并生成板端示例."
    python "${SCRIPT_DIR}/host/upload.py"
    echo "[编译完成] 模型已部署并生成展示示例"
fi
