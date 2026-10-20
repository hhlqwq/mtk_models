#!/usr/bin/env bash
# 容器内一键转换并上传; 板端运行同一入口,默认全量评测.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
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
    export PYTHONPATH="${SCRIPT_DIR}/board${PYTHONPATH:+:${PYTHONPATH}}"
    echo "[板端 1/3] 生成、输入一致性及性能检查: ${EVAL_RUN_ID}."
    python3 "${SCRIPT_DIR}/board/infer.py"
    echo '[板端 2/3] MMLU Accuracy 和 WikiText2 PPL.'
    python3 "${SCRIPT_DIR}/board/evaluate.py"
    echo '[板端 3/3] 汇总同协议指标及部署文件大小.'
    python3 "${SCRIPT_DIR}/board/summarize.py"
else
    : "${MTK_GAI_PYTHON:?请先通过 docker/enter_container.sh 进入容器}"
    : "${MTK_MODELS_ROOT:?缺少公共项目环境}"
    export PYTHONPATH="${MTK_MODELS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    MODEL_BUILD_DIR="$(python3 "${MTK_MODELS_ROOT}/tools/storage.py" --print-build-dir "${SCRIPT_DIR}/../config.yaml")"
    echo '[容器 1/10] 校验官方来源并准备转换副本.'
    "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/prepare_model.py"
    echo '[容器 2/10] 检查评测资源与完整题集上下文需求.'
    "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/evaluate_reference.py" --prepare-only
    echo '[容器 3/10] 生成校准数据.'
    "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/calibrate.py"
    echo '[容器 4/10] 全 W4A16 量化.'
    "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/quantize_model.py"
    echo '[容器 5/10] 合并 Prefill 和 Decode 静态图.'
    "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/shape_model.py"
    echo '[容器 6/10] 编译双 DLA.'
    "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/compile_model.py"
    echo '[容器 7/10] 同协议 FP32 参考评价及板端输入准备.'
    "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/evaluate_reference.py"
    echo '[容器 8/10] 交叉编译公共 Neuron 桥接库.'
    make -C "${SCRIPT_DIR}/board" BUILD_DIR="${MODEL_BUILD_DIR}"
    echo '[容器 9/10] 打包模型、评价输入及板端入口.'
    "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/package_model.py"
    echo '[容器 10/10] 上传开发板.'
    "${MTK_GAI_PYTHON}" "${SCRIPT_DIR}/host/upload.py"
fi
