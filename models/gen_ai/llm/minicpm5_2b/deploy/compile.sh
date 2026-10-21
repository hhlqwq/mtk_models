#!/usr/bin/env bash
# 编译、部署与少量展示示例;正式指标只由 evaluate.sh 生成.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 MTK_EVAL_SCOPE=subset
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
    echo "[错误] 用法: bash compile.sh,不接受参数." >&2
    exit 2
fi
export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
export MTK_RUN_DIR="/tmp/hailongcodex/$(date +%F)/minicpm5_2b_compile_${EVAL_RUN_ID}"

# 仅回显关键节点,完整工具输出保存在日期临时目录.
compile_step() {
    mkdir -p "${MTK_RUN_DIR}"
    COMPILE_STEP_NUMBER=$((COMPILE_STEP_NUMBER + 1))
    local log="${MTK_RUN_DIR}/step_${COMPILE_STEP_NUMBER}.log"
    echo "[编译步骤 ${COMPILE_STEP_NUMBER}] 开始."
    if "$@" >"${log}" 2>&1; then
        echo "[编译步骤 ${COMPILE_STEP_NUMBER}] 完成."
    else
        local status=$?
        echo "[编译失败] 详细日志: ${log}" >&2
        tail -n 30 "${log}" >&2
        return "${status}"
    fi
}
COMPILE_STEP_NUMBER=0
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9 || { echo "[错误] 本模型已有运行任务." >&2; exit 2; }
    CONFIG="${SCRIPT_DIR}/config.json"
    echo "[板端 1/2] 少量输入检查与硬件推理."
    python3 "${SCRIPT_DIR}/board/infer.py" --config "${CONFIG}" --run-id "${EVAL_RUN_ID}"
    echo "[板端 2/2] 生成文本示例已保存至 examples/output/."
    echo "[编译完成] 通路和示例已准备,未生成精度、性能或 RSS 报告."
else
    MODEL_BUILD_DIR="$(python "${MTK_MODELS_ROOT}/tools/storage.py" --print-build-dir "${SCRIPT_DIR}/../config.yaml")"
    mkdir -p "${MODEL_BUILD_DIR}"
    exec 9>"${MODEL_BUILD_DIR}/.run.lock"
    flock -n 9 || { echo '[错误] 本模型已有转换任务.' >&2; exit 2; }
    echo '[容器 1/4] 准备模型、校准、量化并导出双静态图.'
    compile_step python "${SCRIPT_DIR}/host/convert.py"
    echo '[容器 2/4] 编译 Prefill 和 Decode 双 DLA.'
    compile_step python "${SCRIPT_DIR}/host/compile.py"
    echo '[容器 3/4] 交叉编译板端桥接库.'
    compile_step make -C "${SCRIPT_DIR}/board" BUILD_DIR="${MODEL_BUILD_DIR}"
    echo '[容器 4/4] 打包并上传开发板.'
    compile_step python "${SCRIPT_DIR}/host/upload.py"
fi
