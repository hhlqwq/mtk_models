#!/usr/bin/env bash
# 新模型入口骨架: 主机部署,板端默认固定小批量验证.
# 所有模型主机上传后结束,用户登录开发板执行本入口;实现细节见模板 README.
set -euo pipefail
export MTK_OUTPUT_METRICS=1
export MTK_EVAL_SCOPE=subset
export MTK_EVALUATION_WORK=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    export PYTHONDONTWRITEBYTECODE=1
    # 仅 LLM: infer.py 实时显示输入输出并保存文本,移除 render_examples.py 的检查和调用.
    for step in infer.py render_examples.py; do
        [[ -f "${SCRIPT_DIR}/board/${step}" ]] || {
            echo "[未实现] 请先实现板端 ${step}." >&2
            exit 2
        }
    done
    echo "[小批量 1/2] 已部署模型推理."
    python3 "${SCRIPT_DIR}/board/infer.py"
    python3 "${SCRIPT_DIR}/board/compile_accuracy.py"
    echo "[小批量 2/2] 保存 README 所需的展示示例."
    python3 "${SCRIPT_DIR}/board/render_examples.py"
else
    # 直接调用主机命令,输出和错误显示在终端,只补充必要阶段提示.
    # 所有模型上传后提示板端命令,不通过 SSH 自动执行板端推理、示例或评价.
    echo "[未实现] 请实现当前模型的主机导出、量化、编译、打包上传,完成后提示用户登录板端执行 compile.sh." >&2
    exit 2
fi
