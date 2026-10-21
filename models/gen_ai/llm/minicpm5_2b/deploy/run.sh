#!/usr/bin/env bash
# 容器内一键转换并上传; 板端运行同一入口,默认全量评测.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export MTK_EVAL_TEST=0
if (( $# == 1 )) && [[ "$1" == "-test" ]]; then
    export MTK_EVAL_TEST=1
elif (( $# != 0 )); then
    echo '[错误] 用法: bash run.sh [-test]' >&2
    exit 2
fi
export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
if [[ "${MTK_EVAL_TEST}" == 1 && "${EVAL_RUN_ID}" != *_test ]]; then
    export EVAL_RUN_ID="${EVAL_RUN_ID}_test"
fi
if [[ -f "${SCRIPT_DIR}/config.json" ]]; then
    exec 9>"${SCRIPT_DIR}/.run.lock"
    flock -n 9 || { echo '[错误] 本模型已有运行任务.' >&2; exit 2; }
    echo "[板端 1/3] 生成、输入一致性及性能检查: ${EVAL_RUN_ID}."
    python3 "${SCRIPT_DIR}/board/infer.py"
    echo '[板端 2/3] MMLU Accuracy 和 WikiText2 PPL.'
    python3 "${SCRIPT_DIR}/board/evaluate.py"
    echo '[板端 3/3] 汇总同协议指标及部署文件大小.'
    python3 "${SCRIPT_DIR}/board/summarize.py"
else
    MODEL_BUILD_DIR="$(python "${MTK_MODELS_ROOT}/tools/storage.py" --print-build-dir "${SCRIPT_DIR}/../config.yaml")"
    mkdir -p "${MODEL_BUILD_DIR}"
    exec 9>"${MODEL_BUILD_DIR}/.run.lock"
    flock -n 9 || { echo '[错误] 本模型已有转换任务.' >&2; exit 2; }
    echo '[容器 1/5] 准备模型、校准、量化并导出双静态图.'
    python "${SCRIPT_DIR}/host/convert.py"
    echo '[容器 2/5] 编译 Prefill 和 Decode 双 DLA.'
    python "${SCRIPT_DIR}/host/compile.py"
    echo '[容器 3/5] FP32 参考评价及板端输入准备.'
    python "${SCRIPT_DIR}/host/evaluate.py"
    echo '[容器 4/5] 交叉编译板端桥接库.'
    make -C "${SCRIPT_DIR}/board" BUILD_DIR="${MODEL_BUILD_DIR}"
    echo '[容器 5/5] 打包并上传开发板.'
    python "${SCRIPT_DIR}/host/upload.py"
fi
