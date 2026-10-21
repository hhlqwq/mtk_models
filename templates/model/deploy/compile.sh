#!/usr/bin/env bash
# 新模型入口骨架: 主机部署,板端默认固定小批量验证.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 MTK_EVAL_SCOPE=subset
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
    echo "[错误] 用法: bash compile.sh;全量精度使用独立的 evaluate.sh." >&2
    exit 2
fi
export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
export MTK_RUN_DIR="/tmp/hailongcodex/$(date +%F)/model_id_compile_${EVAL_RUN_ID}"
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    CONFIG="${SCRIPT_DIR}/config.json"
    for step in infer.py render_examples.py; do
        [[ -f "${SCRIPT_DIR}/board/${step}" ]] || {
            echo "[未实现] 请先实现板端 ${step}." >&2
            exit 2
        }
    done
    echo "[小批量 1/2] 已部署模型推理."
    python3 "${SCRIPT_DIR}/board/infer.py" --config "${CONFIG}" --run-id "${EVAL_RUN_ID}"
    echo "[小批量 2/2] 保存 README 所需的展示示例."
    python3 "${SCRIPT_DIR}/board/render_examples.py" --config "${CONFIG}" --run-id "${EVAL_RUN_ID}"
else
    # 根据张量契约实现 deploy/host/,工具输出写日期临时日志,终端仅显示关键节点.
    echo "[未实现] 请实现当前模型的主机导出、量化、编译、打包上传和少量示例检查." >&2
    exit 2
fi
