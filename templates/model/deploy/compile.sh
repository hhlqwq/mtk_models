#!/usr/bin/env bash
# 新模型入口骨架: 主机部署,板端默认固定小批量验证.
set -euo pipefail
export MTK_EVAL_SCOPE=subset
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    export PYTHONDONTWRITEBYTECODE=1
    for step in infer.py render_examples.py; do
        [[ -f "${SCRIPT_DIR}/board/${step}" ]] || {
            echo "[未实现] 请先实现板端 ${step}." >&2
            exit 2
        }
    done
    echo "[小批量 1/2] 已部署模型推理."
    python3 "${SCRIPT_DIR}/board/infer.py"
    echo "[小批量 2/2] 保存 README 所需的展示示例."
    python3 "${SCRIPT_DIR}/board/render_examples.py"
else
    # 直接调用主机命令,输出和错误显示在终端,只补充必要阶段提示.
    echo "[未实现] 请实现当前模型的主机导出、量化、编译、打包上传和少量示例检查." >&2
    exit 2
fi
