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
    echo "[板端 2/2] 保存 README 展示示例."
    python3 "${SCRIPT_DIR}/board/render_examples.py"
    echo "[编译完成] 通路和示例已准备,未生成精度、性能或 RSS 报告."
else
    # 编译容器: 各 Python 文件读取 YAML 并直接执行当前步骤.
    echo "[容器 1/4] 导出 FP32 ONNX."
    python "${SCRIPT_DIR}/host/export_model.py"
    echo "[容器 2/4] INT8 校准与量化."
    python "${SCRIPT_DIR}/host/convert_int8.py"
    echo "[容器 3/4] 编译 DLA."
    python "${SCRIPT_DIR}/host/compile_model.py"
    echo "[容器 4/4] 上传到开发板."
    python "${SCRIPT_DIR}/host/upload.py"
    echo "[编译完成] 模型已部署并生成展示示例"
fi
