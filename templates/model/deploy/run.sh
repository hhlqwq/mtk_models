#!/usr/bin/env bash
# 新模型入口骨架: 主机部署,板端默认固定小批量验证.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 MTK_EVAL_SCOPE=subset
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
    echo "[错误] 用法: bash run.sh;全量精度使用独立的 evaluate.sh." >&2
    exit 2
fi
export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
if [[ "${EVAL_RUN_ID}" != *_test ]]; then
    export EVAL_RUN_ID="${EVAL_RUN_ID}_test"
fi
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    CONFIG="${SCRIPT_DIR}/config.json"
    for step in infer.py full_accuracy_board.py; do
        [[ -f "${SCRIPT_DIR}/board/${step}" ]] || {
            echo "[未实现] 请先实现板端 ${step}." >&2
            exit 2
        }
    done
    echo "[小批量 1/2] 已部署模型推理."
    python3 "${SCRIPT_DIR}/board/infer.py" --config "${CONFIG}" --run-id "${EVAL_RUN_ID}"
    echo "[小批量 2/2] 固定子集精度验证."
    python3 "${SCRIPT_DIR}/board/full_accuracy_board.py" --config "${CONFIG}" --run-id "${EVAL_RUN_ID}"
else
    # 根据张量契约实现 deploy/host/,复用公共配置和上传工具后再启用.
    echo "[未实现] 请实现当前模型的主机导出、量化、编译、小批量参考及打包上传步骤." >&2
    exit 2
fi
