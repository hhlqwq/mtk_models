#!/usr/bin/env bash
# 编译、部署与少量展示示例;正式指标只由 evaluate.sh 生成.
set -euo pipefail
export MTK_OUTPUT_METRICS=1
export MTK_EVALUATION_WORK=0
export MTK_EVAL_SCOPE=subset
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    export PYTHONDONTWRITEBYTECODE=1
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9 || { echo "[错误] 本模型已有运行任务." >&2; exit 2; }
    echo "[板端 1/2] 固定子集硬件推理."
    python3 "${SCRIPT_DIR}/board/infer.py"
    echo "[子集精度] 计算任务精度并与同子集浮点参考对比."
    python3 "${SCRIPT_DIR}/board/compile_accuracy.py"
    echo "[板端 2/2] 保存 README 展示示例."
    python3 "${SCRIPT_DIR}/board/render_examples.py"
    echo "[编译完成] 通路和示例已准备,子集精度仅在终端显示,未生成正式精度、性能或 RSS 报告."
else
    : "${MTK_MODELS_ROOT:?请先通过 docker/enter_container.sh 进入容器}"
    echo "[容器 1/4] 导出浮点模型."
    "${MTK_YAMNET_PYTHON:?缺少 YAMNet TensorFlow 环境}" "${SCRIPT_DIR}/host/export_model.py"
    echo "[容器 2/4] 校准与转换."
    python "${SCRIPT_DIR}/host/convert_int8.py"
    echo "[容器 3/4] 编译 DLA."
    python "${SCRIPT_DIR}/host/compile_model.py"
    echo "[容器 4/4] 上传到开发板."
    python "${SCRIPT_DIR}/host/upload.py"
    echo "[小样本数值参考] 仅准备固定少量输入的 CosSim、RMSE、NRMSE 对照."
    python "${MTK_MODELS_ROOT}/tools/prepare_output_reference.py" "${SCRIPT_DIR}"
    python "${MTK_MODELS_ROOT}/tools/compile_accuracy.py" "${SCRIPT_DIR}"
    echo "[编译完成] 模型已上传,请在开发板执行 compile.sh 检查通路并生成展示示例."
fi
